//------------------------------------------------------------------------------
//! @file NetlistVisitor.h
//! @brief An AST visitor (and sub visitors) to construct a netlist
//         representation.
//
// SPDX-FileCopyrightText: Michael Popoloski
// SPDX-License-Identifier: MIT
//------------------------------------------------------------------------------
#pragma once

#include "Debug.h"
#include "Netlist.h"
#include "fmt/core.h"
#include <algorithm>

#include "slang/ast/ASTContext.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/Scope.h"
#include "slang/ast/SemanticFacts.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/ValueSymbol.h"
#include "slang/diagnostics/TextDiagnosticClient.h"
#include "slang/util/Util.h"

using namespace slang;

namespace netlist {

static std::string getSymbolHierPath(const ast::Symbol& symbol) {
    std::string buffer;
    symbol.getHierarchicalPath(buffer);
    return buffer;
}

static void connectDeclToVar(Netlist& netlist, NetlistNode& declNode,
                             const std::string& hierarchicalPath) {
    auto* varNode = netlist.lookupVariable(hierarchicalPath);
    netlist.addEdge(*varNode, declNode);
    DEBUG_PRINT("Edge decl {} to ref {}\n", varNode->getName(), declNode.getName());
}

static void connectVarToDecl(Netlist& netlist, NetlistNode& varNode,
                             const std::string& hierarchicalPath) {
    auto* declNode = netlist.lookupVariable(hierarchicalPath);
    netlist.addEdge(varNode, *declNode);
    DEBUG_PRINT("Edge ref {} to decl {}\n", varNode.getName(), declNode->getName());
}

static void connectVarToVar(Netlist& netlist, NetlistNode& sourceVarNode,
                            NetlistNode& targetVarNode) {
    netlist.addEdge(sourceVarNode, targetVarNode);
    DEBUG_PRINT("Edge ref {} to ref {}\n", sourceVarNode.getName(), targetVarNode.getName());
}

/// An AST visitor to identify variable references with selectors in
/// expressions, adding them to a visit list during the traversal.
class VariableReferenceVisitor : public ast::ASTVisitor<VariableReferenceVisitor, false, true> {
public:
    explicit VariableReferenceVisitor(Netlist& netlist, ast::EvalContext& evalCtx,
                                      bool leftOperand = false) :
        netlist(netlist), evalCtx(evalCtx), leftOperand(leftOperand) {}

    void handle(const ast::NamedValueExpression& expr) {

        // If the symbol reference is to a constant (eg a parameter or enum
        // value), then skip it.
        if (!expr.eval(evalCtx).bad()) {
            return;
        }

        // Add the variable reference to the netlist.
        auto& node = netlist.addVariableReference(expr.symbol, expr, leftOperand);
        varList.push_back(&node);
        for (auto* selector : selectors) {
            if (selector->kind == ast::ExpressionKind::ElementSelect) {
                const auto& expr = selector->as<ast::ElementSelectExpression>();
                auto index = expr.selector().eval(evalCtx);
                node.addElementSelect(expr, index);
            }
            else if (selector->kind == ast::ExpressionKind::RangeSelect) {
                const auto& expr = selector->as<ast::RangeSelectExpression>();
                auto leftIndex = expr.left().eval(evalCtx);
                auto rightIndex = expr.right().eval(evalCtx);
                node.addRangeSelect(expr, leftIndex, rightIndex);
            }
            else if (selector->kind == ast::ExpressionKind::MemberAccess) {
                node.addMemberAccess(selector->as<ast::MemberAccessExpression>().member.name);
            }
        }

        // Reverse the selectors.
        std::reverse(node.selectors.begin(), node.selectors.end());

        // Determine the access range to the variable.
        if (!selectors.empty()) {
            SmallVector<std::pair<const slang::ast::ValueSymbol*, const slang::ast::Expression*>>
                prefixes;
            selectors.front()->getLongestStaticPrefixes(prefixes, evalCtx);
            SLANG_ASSERT(prefixes.size() == 1);
            auto [prefixSymbol, prefixExpr] = prefixes.back();
            auto bounds = slang::ast::ValueDriver::getBounds(*prefixExpr, evalCtx,
                                                             prefixSymbol->getType());
            node.bounds = {static_cast<int32_t>(bounds->first),
                           static_cast<int32_t>(bounds->second)};
        }
        else {
            node.bounds = {0, getTypeBitWidth(expr.symbol.getType()) - 1};
        }
        DEBUG_PRINT("Variable reference: {} bounds {}:{}\n", node.toString(), node.bounds.lower(),
                    node.bounds.upper());

        // Clear the selectors for the next named value.
        selectors.clear();
    }

    void handle(const ast::ElementSelectExpression& expr) {
        selectors.push_back(&expr);
        expr.value().visit(*this);
    }

    void handle(const ast::RangeSelectExpression& expr) {
        selectors.push_back(&expr);
        expr.value().visit(*this);
    }

    void handle(const ast::MemberAccessExpression& expr) {
        selectors.push_back(&expr);
        expr.value().visit(*this);
    }

    std::vector<NetlistNode*>& getVars() { return varList; }

private:
    Netlist& netlist;
    ast::EvalContext& evalCtx;
    /// Whether this traversal is the target of an assignment or not.
    bool leftOperand;
    std::vector<NetlistNode*> varList;
    std::vector<const ast::Expression*> selectors;

    std::pair<size_t, size_t> getTypeBitWidthImpl(slang::ast::Type const& type) {
        size_t fixedSize = type.getBitWidth();
        if (fixedSize > 0) {
            return {1, fixedSize};
        }

        size_t multiplier = 0;
        const auto& ct = type.getCanonicalType();
        if (ct.kind == slang::ast::SymbolKind::FixedSizeUnpackedArrayType) {
            auto [multiplierElem, fixedSizeElem] = getTypeBitWidthImpl(*type.getArrayElementType());
            auto rw = ct.as<slang::ast::FixedSizeUnpackedArrayType>().range.width();
            return {multiplierElem * rw, fixedSizeElem};
        }

        SLANG_UNREACHABLE;
    }

    /// Return the bit width of a slang type, treating unpacked arrays as
    /// as if they were packed.
    int32_t getTypeBitWidth(slang::ast::Type const& type) {
        auto [multiplierElem, fixedSizeElem] = getTypeBitWidthImpl(type);
        return (int32_t)(multiplierElem * fixedSizeElem);
    }
};

/// An AST visitor to create dependencies between occurrances of variables
/// appearing on the left and right hand sides of assignment statements.
class AssignmentVisitor : public ast::ASTVisitor<AssignmentVisitor, false, true> {
public:
    explicit AssignmentVisitor(Netlist& netlist, ast::EvalContext& evalCtx,
                               SmallVector<NetlistNode*>& condVars) :
        netlist(netlist), evalCtx(evalCtx), condVars(condVars) {}

    void handle(const ast::AssignmentExpression& expr) {
        // Collect variable references on the left-hand side of the assignment.
        VariableReferenceVisitor visitorLHS(netlist, evalCtx, true);
        expr.left().visit(visitorLHS);
        // Collect variable references on the right-hand side of the assignment.
        VariableReferenceVisitor visitorRHS(netlist, evalCtx, false);
        expr.right().visit(visitorRHS);
        for (auto* leftNode : visitorLHS.getVars()) {
            // Add edge from LHS variable refrence to variable declaration.
            connectVarToDecl(netlist, *leftNode, getSymbolHierPath(leftNode->symbol));
            for (auto* rightNode : visitorRHS.getVars()) {
                // Add edge from variable declaration to RHS variable reference.
                connectDeclToVar(netlist, *rightNode, getSymbolHierPath(rightNode->symbol));
                // Add edge from RHS expression term to LHS expression terms.
                connectVarToVar(netlist, *rightNode, *leftNode);
            }
        }
        for (auto* condNode : condVars) {
            // Add edge from conditional variable declaraiton to the reference.
            connectDeclToVar(netlist, *condNode, getSymbolHierPath(condNode->symbol));
            for (auto* leftNode : visitorLHS.getVars()) {
                // Add edge from conditional variable to the LHS variable.
                connectVarToVar(netlist, *condNode, *leftNode);
            }
        }
    }

private:
    Netlist& netlist;
    ast::EvalContext& evalCtx;
    SmallVector<NetlistNode*>& condVars;
};

/// An AST visitor for proceural blocks that performs loop unrolling.
class ProceduralBlockVisitor : public ast::ASTVisitor<ProceduralBlockVisitor, true, false> {
public:
    bool anyErrors = false;

    explicit ProceduralBlockVisitor(ast::Compilation& compilation, Netlist& netlist) :
        netlist(netlist),
        evalCtx(ast::ASTContext(compilation.getRoot(), ast::LookupLocation::max)) {
        evalCtx.pushEmptyFrame();
    }

    void handle(const ast::ForLoopStatement& loop) {

        // Conditions this loop cannot be unrolled.
        if (loop.loopVars.empty() || !loop.stopExpr || loop.steps.empty() || anyErrors) {
            loop.body.visit(*this);
            return;
        }

        // Attempt to unroll the loop. If we are unable to collect constant values
        // for all loop variables across all iterations, we won't unroll at all.
        auto handleFail = [&] {
            for (auto var : loop.loopVars) {
                evalCtx.deleteLocal(var);
            }
            loop.body.visit(*this);
        };

        // Create a list of the initialised loop variables.
        SmallVector<ConstantValue*> localPtrs;
        for (auto var : loop.loopVars) {
            auto init = var->getInitializer();
            if (!init) {
                handleFail();
                return;
            }

            auto cv = init->eval(evalCtx);
            if (!cv) {
                handleFail();
                return;
            }

            localPtrs.push_back(evalCtx.createLocal(var, std::move(cv)));
        }

        // Create a list of all the loop variable values across all iterations.
        SmallVector<ConstantValue, 16> values;
        while (true) {
            auto cv = step() ? loop.stopExpr->eval(evalCtx) : ConstantValue();
            if (!cv) {
                handleFail();
                return;
            }

            if (!cv.isTrue()) {
                break;
            }

            for (auto local : localPtrs) {
                values.emplace_back(*local);
            }

            for (auto step : loop.steps) {
                if (!step->eval(evalCtx)) {
                    handleFail();
                    return;
                }
            }
        }

        // We have all the loop iteration values. Go back through
        // and visit the loop body for each iteration.
        for (size_t i = 0; i < values.size();) {
            for (auto local : localPtrs) {
                *local = std::move(values[i++]);
            }

            loop.body.visit(*this);
            if (anyErrors) {
                return;
            }
        }
    }

    void handle(const ast::ConditionalStatement& stmt) {
        // Evaluate the condition; if not constant visit both sides (the
        // fallback option), otherwise visit only the side that matches the
        // condition.

        auto fallback = [&] {
            // Create a list of variables appearing in the condition
            // expression.
            VariableReferenceVisitor varRefVisitor(netlist, evalCtx);
            for (auto& cond : stmt.conditions) {
                if (cond.pattern) {
                    // Skip.
                    continue;
                }
                cond.expr->visit(varRefVisitor);
            }

            // Push the condition variables.
            for (auto& varRef : varRefVisitor.getVars()) {
                condVarsStack.push_back(varRef);
            }

            // Visit the 'then' and 'else' statements, whose execution is
            // under the control of the condition variables.
            stmt.ifTrue.visit(*this);
            if (stmt.ifFalse) {
                stmt.ifFalse->visit(*this);
            }

            // Pop the conditon variables.
            for (auto& varRef : varRefVisitor.getVars()) {
                condVarsStack.pop_back();
            }
        };

        for (auto& cond : stmt.conditions) {
            if (cond.pattern || !step()) {
                fallback();
                return;
            }

            auto result = cond.expr->eval(evalCtx);
            if (!result) {
                fallback();
                return;
            }

            if (!result.isTrue()) {
                if (stmt.ifFalse) {
                    stmt.ifFalse->visit(*this);
                }
                return;
            }
        }

        stmt.ifTrue.visit(*this);
    }

    void handle(const ast::ExpressionStatement& stmt) {
        step();
        AssignmentVisitor visitor(netlist, evalCtx, condVarsStack);
        stmt.visit(visitor);
    }

private:
    bool step() {
        if (anyErrors || !evalCtx.step(SourceLocation::NoLocation)) {
            anyErrors = true;
            return false;
        }
        return true;
    }

    Netlist& netlist;
    ast::EvalContext evalCtx;
    SmallVector<NetlistNode*> condVarsStack;
};

/// Hold the information relating to a port contained within an interface port
/// of an instance in order that it can be flattened an appear as a regular
/// port in the netlist.
class FlatInterfacePort {
public:
  ast::InstanceSymbol const &instance;
  ast::InterfacePortSymbol const &iface;
  ast::Symbol const &port;
  ast::ArgumentDirection direction;
  std::ModportSymbol &modport;

  FlatInterfacePort(ast::InstanceSymbol const& instance, ast::InterfacePortSymbol const& iface,
                    ast::Symbol const& port) :
      instance(instance),
      iface(iface), port(port) {}

  /// Return the effective hierarchical path for this flattened port.
  auto getHierarchicalPath() -> std::string {
   return fmt::format("{}.{}.{}", getSymbolHierPath(instance), iface.name, port.name);
  }
};

/// A visitor that traverses the AST and builds a netlist representation.
class NetlistVisitor : public ast::ASTVisitor<NetlistVisitor, true, false> {
public:
    explicit NetlistVisitor(ast::Compilation& compilation, Netlist& netlist) :
        compilation(compilation), netlist(netlist) {}

    /// Connect the ports of a module instance to the variables that connect to
    /// it in the parent scope. Given a port hookup of the form:
    ///
    ///   .foo(expr(x, y))
    ///
    /// Where expr() is an expression involving some variables.
    ///
    /// Then, add the following edges:
    ///
    /// - Input port:
    ///
    ///   var decl x -> var ref x -> port var ref foo
    ///
    /// - Output port:
    ///
    ///   var decl y <- var ref y <- port var ref foo
    ///
    /// - InOut port:
    ///
    ///   var decl x -> var ref x -> port var ref foo
    ///   var decl y <- var ref y <- port var ref foo
    void connectPortExternal(NetlistNode* node, ast::Symbol const& portSymbol,
                             ast::ArgumentDirection direction) {
        switch (direction) {
            case ast::ArgumentDirection::In:
                connectDeclToVar(netlist, *node, getSymbolHierPath(node->symbol));
                connectVarToDecl(netlist, *node, getSymbolHierPath(portSymbol));
                break;
            case ast::ArgumentDirection::Out:
                connectDeclToVar(netlist, *node, getSymbolHierPath(portSymbol));
                connectVarToDecl(netlist, *node, getSymbolHierPath(node->symbol));
                break;
            case ast::ArgumentDirection::InOut:
                connectDeclToVar(netlist, *node, getSymbolHierPath(node->symbol));
                connectDeclToVar(netlist, *node, getSymbolHierPath(portSymbol));
                connectVarToDecl(netlist, *node, getSymbolHierPath(node->symbol));
                connectVarToDecl(netlist, *node, getSymbolHierPath(portSymbol));
                break;
            case ast::ArgumentDirection::Ref:
                break;
        }
    }

    /// Connect the ports of a module instance to their corresponding variables
    /// occuring in the body of the module.
    void connectPortInternal(NetlistNode& port) {
        if (auto* internalSymbol = port.symbol.as<ast::PortSymbol>().internalSymbol) {
            std::string pathBuffer;
            internalSymbol->getHierarchicalPath(pathBuffer);
            auto* variableNode = netlist.lookupVariable(pathBuffer);
            switch (port.symbol.as<ast::PortSymbol>().direction) {
                case ast::ArgumentDirection::In:
                    netlist.addEdge(port, *variableNode);
                    break;
                case ast::ArgumentDirection::Out:
                    netlist.addEdge(*variableNode, port);
                    break;
                case ast::ArgumentDirection::InOut:
                    netlist.addEdge(port, *variableNode);
                    netlist.addEdge(*variableNode, port);
                    break;
                case ast::ArgumentDirection::Ref:
                    break;
            }
        }
        else {
            SLANG_UNREACHABLE;
        }
    }

    /// Given an interface port symbol, return a list of ports defined by the
    /// interface instance and any modport specifier.
    auto getFlatIfacePorts(ast::InstanceSymbol const &instance, ast::InterfacePortSymbol const& ifacePort)
        -> SmallVector<FlatInterfacePort> {

        // Get the interface instance.
        const ast::Symbol* conn;
        const ast::ModportSymbol* modport = nullptr;
        std::tie(conn, modport) = ifacePort.getConnection();
        if (!conn) {
          // Bad.
          return {};
        }

        // Unwrap any array dimensions.
        SmallVector<ConstantRange, 4> dims;
        auto origSymbol = conn;
        while (conn->kind == ast::SymbolKind::InstanceArray) {
            auto& array = conn->as<ast::InstanceArraySymbol>();
            if (array.elements.empty()) {
              // Bad.
              return {};
            }

            dims.push_back(array.range);
            conn = array.elements[0];
        }

        // Get the interface instance body.
        const ast::InstanceBodySymbol* iface = nullptr;
        if (conn->kind == ast::SymbolKind::Modport) {
            modport = &conn->as<ast::ModportSymbol>();
            iface = &conn->getParentScope()->asSymbol().as<ast::InstanceBodySymbol>();
        }
        else {
            iface = &conn->as<ast::InstanceSymbol>().body;
        }

        // Ports
        if (modport) {
          DEBUG_PRINT("Interface modport: {}\n", modport->name);
        }

        SmallVector<FlatInterfacePort, 8> ports;

        // Add all interface members to the list of flattened ports.
        for (auto& member : iface->members()) {
          std::string path;
          member.getHierarchicalPath(path);
          if (member.kind != ast::SymbolKind::Modport) {
              DEBUG_PRINT("Interface member: instance={} port={} name={} path={}\n", instance.name,
                          ifacePort.name, member.name, path);
              ports.emplace_back(instance, ifacePort, member, modport);
          }
        }

        // Apply modport direction constraints.
        for (auto& member : iface->members()) {
          std::string path;
          member.getHierarchicalPath(path);
          if (member.kind == ast::SymbolKind::Modport && member.name == modport->name) {
            DEBUG_PRINT("Interface modport: instance={} port={} name={} path={}\n", instance.name,
                        ifacePort.name, member.name, path);
            for (auto& modportMember : member.as<ast::ModportSymbol>().members()) {
                    if (modportMember.kind == ast::SymbolKind::ModportPort) {
                        auto& port = modportMember.as<ast::ModportPortSymbol>();
                        DEBUG_PRINT("Modport port member: name={} direction={}\n", port.name,
                                    int(port.direction));
                        auto it = std::find_if(ports.begin(), ports.end(),
                                            [&port](FlatInterfacePort const& p) {
                                                return p.port.name == port.name;
                                            });
                        if (it != ports.end()) {
                            it->direction = port.direction;
                        }
                        else {
                            SLANG_ASSERT(0 && "modport name not found in interface");
                        }
                    }
            }
          }
        }

        return ports;
    }

    auto handleInstanceVariables(ast::InstanceSymbol const &symbol) {
      // Add variable declarations.
      for (auto& member : symbol.body.members()) {
          if (member.kind == ast::SymbolKind::Variable || member.kind == ast::SymbolKind::Net) {
            netlist.addVariableDeclaration(member);
          }
      }
    }

    /// Handle making connections form port members to internal variable
    /// declarations. This must be called before 'handleInstanceExtPortConn'
    /// becuase it creates the port declarations in the netlist that are
    /// connected to externally.
    auto handleInstanceIntPortConn(ast::InstanceSymbol const &symbol) {

        for (auto& member : symbol.body.members()) {
            if (member.kind == ast::SymbolKind::Port) {
                // Create the port declaration netlist node and connect it to
                // the corresponding local variable declaration.
                auto& portNode = netlist.addPortDeclaration(member);
                connectPortInternal(portNode);
            }
            else if (member.kind == ast::SymbolKind::MultiPort) {
              // TODO
              assert(0 && "unimplemented");
            }
            else if (member.kind == ast::SymbolKind::InterfacePort) {
                //auto& ifacePort = member.as<ast::InterfacePortSymbol>();
                //auto ifacePortList = getFlatIfacePorts(symbol, ifacePort);

                //// Create port and variable declarations for each flattened member of the
                //// interface.
                //for (auto &flatPort : ifacePortList) {
                //  auto hierPath = flatPort.getHierarchicalPath();
                //  auto& portDecl = netlist.addPortDeclaration(flatPort.port, hierPath);
                //  auto& varDecl = netlist.addVariableDeclaration(flatPort.port, hierPath);

                //  // Connect the port declaration to internal reference(s) to
                //  // the name.
                //  auto* variableNode = netlist.lookupVariable(hierPath);
                //  switch (flatPort.direction) {
                //      case ast::ArgumentDirection::In:
                //          netlist.addEdge(portDecl, varDecl);
                //          break;
                //      case ast::ArgumentDirection::Out:
                //          netlist.addEdge(varDecl, portDecl);
                //          break;
                //      case ast::ArgumentDirection::InOut:
                //          netlist.addEdge(portDecl, varDecl);
                //          netlist.addEdge(varDecl, portDecl);
                //          break;
                //      case ast::ArgumentDirection::Ref:
                //          break;
                //  }
                //}
            }
        }
    }

    // Handle making connections from the port connections to the port
    // declarations of an instance.
    auto handleInstanceExtPortConn(ast::InstanceSymbol const &symbol) {

        for (auto* portConnection : symbol.getPortConnections()) {

            if (portConnection->port.kind == ast::SymbolKind::Port) {
                auto& port = portConnection->port.as<ast::PortSymbol>();
                auto direction = portConnection->port.as<ast::PortSymbol>().direction;

                ast::EvalContext evalCtx(
                  ast::ASTContext(compilation.getRoot(), ast::LookupLocation::max));

                // The port is the target of an assignment if it is an input.
                bool isLeftOperand = direction == ast::ArgumentDirection::In ||
                                     direction == ast::ArgumentDirection::InOut;

                // Collect variable references in the port expression.
                VariableReferenceVisitor visitor(netlist, evalCtx, isLeftOperand);
                portConnection->getExpression()->visit(visitor);

                for (auto* node : visitor.getVars()) {
                  connectPortExternal(node, portConnection->port, direction);
                }
            }
            else if (portConnection->port.kind == ast::SymbolKind::MultiPort) {
                auto& port = portConnection->port.as<ast::MultiPortSymbol>();
                // TODO
                assert(0 && "unimplemented");
            }
            else if (portConnection->port.kind == ast::SymbolKind::InterfacePort) {
                //auto& ifacePort = portConnection->port.as<ast::InterfacePortSymbol>();
                //auto ifacePortList = getFlatIfacePorts(symbol, ifacePort);

                //for (auto &flatPort : ifacePortList) {
                //    auto* portNode = netlist.lookupVariable(getSymbolHierPath(flatPort.port));
                //    // TODO
                //    //connectPortExternal(portNode, , flatPort.direction);
                //}

            } else {
              SLANG_UNREACHABLE;
            }
        }
    }

    /// Variable declaration.
    void handle(const ast::VariableSymbol& symbol) {}

    /// Net declaration.
    void handle(const ast::NetSymbol& symbol) {}

    /// Port declaration.
    void handle(const ast::PortSymbol& symbol) {}

    /// Instance.
    void handle(const ast::InstanceSymbol& symbol) {
        DEBUG_PRINT("Instance {}\n", symbol.name);

        if (symbol.name.empty()) {
            // An instance without a name has been excluded from the design.
            // This can happen when the --top option is used and there is an
            // uninstanced module.
            return;
        }

        handleInstanceVariables(symbol);
        handleInstanceIntPortConn(symbol);
        handleInstanceExtPortConn(symbol);

        symbol.body.visit(*this);
    }

    /// Procedural block.
    void handle(const ast::ProceduralBlockSymbol& symbol) {
        ProceduralBlockVisitor visitor(compilation, netlist);
        symbol.visit(visitor);
    }

    /// Continuous assignment statement.
    void handle(const ast::ContinuousAssignSymbol& symbol) {
        ast::EvalContext evalCtx(ast::ASTContext(compilation.getRoot(), ast::LookupLocation::max));
        SmallVector<NetlistNode*> condVars;
        AssignmentVisitor visitor(netlist, evalCtx, condVars);
        symbol.visit(visitor);
    }

private:
    ast::Compilation& compilation;
    Netlist& netlist;
};

} // namespace netlist
