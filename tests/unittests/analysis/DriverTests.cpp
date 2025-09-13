// SPDX-FileCopyrightText: Michael Popoloski
// SPDX-License-Identifier: MIT

#include "AnalysisTests.h"

TEST_CASE("Class method driver crash regress GH #552") {
    auto& code = R"(
interface I;
    logic l;
    modport mst ( output l );
    modport slv ( input l );
endinterface

module m(I.slv i);
    logic x;
    assign x = i.l;
endmodule

module n(I.mst i);
    assign i.l = 1 ;
endmodule 

module top;
    I i();
    m u_m(i);
    n u_n(i);
endmodule
)";

    Compilation compilation;
    AnalysisManager analysisManager;
    auto [diags, design] = analyze(code, compilation, analysisManager);
    CHECK_DIAGS_EMPTY;
}
