#pragma once

namespace slang {

enum class DiagCode : uint8_t {
    NonPrintableChar,
    UTF8Char,
    UnicodeBOM,
    EmbeddedNull,
    MisplacedDirectiveChar,
    EscapedWhitespace,
    NewlineInStringLiteral,
    UnterminatedStringLiteral,
    UnterminatedBlockComment,
    SplitBlockCommentInDirective,
    MissingExponentDigits,
    MissingFractionalDigits,
    OctalEscapeCodeTooBig,
    InvalidHexEscapeCode,
    UnknownEscapeCode,
    RealExponentTooLarge,
    SignedLiteralTooLarge,
    IntegerSizeZero,
    IntegerSizeTooLarge,
    MissingVectorBase,
    MissingVectorDigits
};

class SyntaxError {
public:
    DiagCode code;
    int location;
    int width;

    SyntaxError(DiagCode code, int location, int width)
        : code(code), location(location), width(width) {
    }
};

class Diagnostics {
public:
    Diagnostics();

    bool empty() const { return syntaxErrors.empty(); }

    void clear() { syntaxErrors.clear(); }
    void add(const SyntaxError& error);

    // TODO: this is temporary
    const SyntaxError& last() { return *(syntaxErrors.end() - 1); }

private:
    Buffer<SyntaxError> syntaxErrors;
};

}