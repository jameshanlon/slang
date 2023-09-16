//------------------------------------------------------------------------------
//! @file VariableSelectorsTests.cpp
//! @brief Tests for handling of variable selectors.
//
// SPDX-FileCopyrightText: Michael Popoloski
// SPDX-License-Identifier: MIT
//------------------------------------------------------------------------------

#include "NetlistTest.h"
#include "SplitVariables.h"
#include <stdexcept>


/// Helper method to extract a variable reference from a netlist and return the
/// bit range associated with it.
ConstantRange getBitRange(Netlist &netlist, std::string_view variableSyntax) {
    auto* node = netlist.lookupVariableReference(variableSyntax);
    if (node == nullptr) {
      throw std::runtime_error(fmt::format("Could not find node {}", variableSyntax));
    }
    return AnalyseVariableReference::create(*node).getBitRange();
}

//===---------------------------------------------------------------------===//
// Scalar selectors.
//===---------------------------------------------------------------------===//

TEST_CASE("Scalar element and range") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  int foo;
  always_comb begin
    foo = 0;
    foo[0] = 0;
    foo[1] = 0;
    foo[7:7] = 0;
    foo[1:0] = 0;
    foo[3:1] = 0;
    foo[7:4] = 0;
    foo[3:1][2:1] = 0;
    foo[7:4][6:5] = 0;
    foo[3:1][2:1][1] = 0;
    foo[7:4][6:5][5] = 0;
    foo[a] = 0;
    foo[a+:1] = 0;
    foo[a-:1] = 0;
    foo[a+:1][a] = 0;
    foo[a-:1][a] = 0;
    foo[a+:1][a-:1] = 0;
    foo[a+:1][a-:1][a] = 0;
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[0]") == ConstantRange(0, 0));
    CHECK(getBitRange(netlist, "foo[1]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[7:7]") == ConstantRange(7, 7));
    CHECK(getBitRange(netlist, "foo[1:0]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[3:1]") == ConstantRange(1, 3));
    CHECK(getBitRange(netlist, "foo[7:4]") == ConstantRange(4, 7));
    CHECK(getBitRange(netlist, "foo[3:1][2:1]") == ConstantRange(1, 2));
    CHECK(getBitRange(netlist, "foo[7:4][6:5]") == ConstantRange(5, 6));
    CHECK(getBitRange(netlist, "foo[3:1][2:1][1]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[7:4][6:5][5]") == ConstantRange(5, 5));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a-:1]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a+:1][a]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a-:1][a]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a+:1][a-:1]") == ConstantRange(0, 31));
    CHECK(getBitRange(netlist, "foo[a+:1][a-:1][a]") == ConstantRange(0, 31));
}

//===---------------------------------------------------------------------===//
// Packed array selectors.
//===---------------------------------------------------------------------===//

TEST_CASE("Packed 1D array element and range") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic [3:0] foo;
  always_comb begin
    foo = 0;
    foo[0] = 0;
    foo[1] = 0;
    foo[2] = 0;
    foo[3] = 0;
    foo[3:3] = 0;
    foo[1:0] = 0;
    foo[3:0] = 0;
    foo[2:1] = 0;
    foo[3:1][2:1][1] = 0;
    foo[a] = 0;
    foo[a+:1] = 0;
    foo[a-:1] = 0;
    foo[a+:1][a] = 0;
    foo[a-:1][a] = 0;
    foo[a+:1][a-:1] = 0;
    foo[a+:1][a-:1][a] = 0;
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[0]") == ConstantRange(0, 0));
    CHECK(getBitRange(netlist, "foo[1]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[2]") == ConstantRange(2, 2));
    CHECK(getBitRange(netlist, "foo[3]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[3:3]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[1:0]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[3:0]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[2:1]") == ConstantRange(1, 2));
    CHECK(getBitRange(netlist, "foo[3:1][2:1][1]") == ConstantRange(1, 1));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a-:1]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a+:1][a]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a-:1][a]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a+:1][a-:1]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a+:1][a-:1][a]") == ConstantRange(0, 3));
}

TEST_CASE("Packed 1D array element and range non-zero indexed") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic [7:4] foo;
  always_comb begin
    foo = 0;
    foo[4] = 0;
    foo[5] = 0;
    foo[6] = 0;
    foo[7] = 0;
    foo[7:7] = 0;
    foo[5:4] = 0;
    foo[7:4] = 0;
    foo[6:5] = 0;
    foo[7:4][6:5][5] = 0;
    foo[a] = 0;
    foo[a+:1] = 0;
    foo[a-:1] = 0;
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[4]") == ConstantRange(0, 0));
    CHECK(getBitRange(netlist, "foo[5]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[6]") == ConstantRange(2, 2));
    CHECK(getBitRange(netlist, "foo[7]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[7:7]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[5:4]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[7:4]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[6:5]") == ConstantRange(1, 2));
    CHECK(getBitRange(netlist, "foo[7:4][6:5][5]") == ConstantRange(1, 1));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[a-:1]") == ConstantRange(0, 3));
}

TEST_CASE("Packed 2D array element and range") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic [3:0] [1:0] foo;
  always_comb begin
    foo = 0;
    foo[0] = 0;
    foo[1] = 0;
    foo[2] = 0;
    foo[3] = 0;
    foo[0][1] = 0;
    foo[1][1] = 0;
    foo[2][1] = 0;
    foo[3][1] = 0;
    foo[1:0] = 0;
    foo[3:2] = 0;
    foo[3:0][2:1] = 0;
    foo[3:0][2:1][1] = 0;
    foo[a] = 0;
    foo[a][1] = 0;
    foo[a][a] = 0;
    foo[a+:1] = 0;
    foo[a-:1] = 0;
    //foo[a+:1][1] = 0;
    //foo[a-:1][1] = 0;
    foo[1][a] = 0;
    foo[1][a+:1] = 0;
    foo[1][a-:1] = 0;
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[0]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[1]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[2]") == ConstantRange(4, 5));
    CHECK(getBitRange(netlist, "foo[3]") == ConstantRange(6, 7));
    CHECK(getBitRange(netlist, "foo[0][1]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[1][1]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[2][1]") == ConstantRange(5, 5));
    CHECK(getBitRange(netlist, "foo[3][1]") == ConstantRange(7, 7));
    CHECK(getBitRange(netlist, "foo[1:0]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[3:2]") == ConstantRange(4, 7));
    CHECK(getBitRange(netlist, "foo[3:0][2:1]") == ConstantRange(2, 5));
    CHECK(getBitRange(netlist, "foo[3:0][2:1][1]") == ConstantRange(2, 3));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a][1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a][a]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a-:1]") == ConstantRange(0, 7));
    //CHECK(getBitRange(netlist, "foo[a+:1][1]") == ConstantRange(0, 7));
    //CHECK(getBitRange(netlist, "foo[a-:1][1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[1][a]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[1][a+:1]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[1][a-:1]") == ConstantRange(2, 3));
}

TEST_CASE("Packed 2D array element and range, non-zero indexing") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic [7:4] [3:2] foo;
  always_comb begin
    foo = 0;
    foo[4] = 0;
    foo[4][3] = 0;
    foo[5:4] = 0;
    foo[7:4][6:5] = 0;
    foo[7:5][6:5][5] = 0;
    foo[a] = 0;
    foo[a+:1] = 0;
    foo[a-:1] = 0;
    foo[5][a] = 0;
    foo[5][a+:1] = 0;
    foo[5][a-:1] = 0;
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[4]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[4][3]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[5:4]") == ConstantRange(0, 3));
    CHECK(getBitRange(netlist, "foo[7:4][6:5]") == ConstantRange(2, 5));
    CHECK(getBitRange(netlist, "foo[7:5][6:5][5]") == ConstantRange(2, 3));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a-:1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[5][a]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[5][a+:1]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[5][a-:1]") == ConstantRange(2, 3));
}

//===---------------------------------------------------------------------===//
// Unpacked array selectors.
//===---------------------------------------------------------------------===//

TEST_CASE("Unpacked 1D array element") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic foo [1:0];
  logic bar [1:0];
  always_comb begin
    foo = bar;
    foo[0] = 0;
    foo[1] = 0;
    foo[a] = 0;
    foo[a+:1] = '{0};
    foo[a-:2] = '{0, 0};
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[0]") == ConstantRange(0, 0));
    CHECK(getBitRange(netlist, "foo[1]") == ConstantRange(1, 1));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[a-:2]") == ConstantRange(0, 1));
}

TEST_CASE("Unpacked 2D array element and range") {
    auto tree = SyntaxTree::fromText(R"(
module m (input int a);
  logic foo [3:0] [1:0];
  logic bar [1:0];
  always_comb begin
    foo[0] = bar;
    foo[1] = bar;
    foo[2] = bar;
    foo[3] = bar;
    foo[0][1] = 0;
    foo[1][1] = 0;
    foo[2][1] = 0;
    foo[3][1] = 0;
    foo[a] = bar;
    foo[a][1] = 0;
    foo[a][a] = 0;
    foo[a+:1] = '{'{0, 0}};
    foo[a-:2] = '{'{0, 0}, '{0, 0}};
    //foo[a+:1][1] = 0;
    //foo[a-:1][1] = 0;
    foo[1][a] = 0;
    foo[1][a+:1] = '{0};
    foo[1][a-:2] = '{0, 0};
  end
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    NO_COMPILATION_ERRORS;
    auto netlist = createNetlist(compilation);
    CHECK(getBitRange(netlist, "foo[0]") == ConstantRange(0, 1));
    CHECK(getBitRange(netlist, "foo[1]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[0][1]") == ConstantRange(1, 1));
    CHECK(getBitRange(netlist, "foo[1][1]") == ConstantRange(3, 3));
    CHECK(getBitRange(netlist, "foo[2][1]") == ConstantRange(5, 5));
    CHECK(getBitRange(netlist, "foo[3][1]") == ConstantRange(7, 7));
    // Dynamic indices.
    CHECK(getBitRange(netlist, "foo[a]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a][1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a][a]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a+:1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[a-:2]") == ConstantRange(0, 7));
    //CHECK(getBitRange(netlist, "foo[a+:1][1]") == ConstantRange(0, 7));
    //CHECK(getBitRange(netlist, "foo[a-:1][1]") == ConstantRange(0, 7));
    CHECK(getBitRange(netlist, "foo[1][a]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[1][a+:1]") == ConstantRange(2, 3));
    CHECK(getBitRange(netlist, "foo[1][a-:2]") == ConstantRange(2, 3));
}

