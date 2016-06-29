from symbol import dotted_name
class AstNode():
    members=["attributes", "token_span", "children"]
    def __init__(self, **kw):
        all_members = set(sum(cls.members for cls in self.__class__.mro() if hasattr(cls, "members")))
        assert set(kw.keys).issubset(all_members), kw.keys - all_members
        self.__dict__.update(kw)
        if not hasattr(self, "children"):
            self.children = None

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join("%s=%r" % (k, v) for k, v in self.__dict__.items()))

    def show(self, nesting=0):
        attrs = ", ".join("%s=%r" % (k, v) for k, v in self.__dict__.items() if k != "children")
        print("%s%s(%s)" % ("  " * nesting, self.__class__.__name__, attrs))
        for child in self.children:
            child.show(nesting+1)


class Module(AstNode): members = ["name"]
class VarDecl(AstNode): pass
class VarRef(AstNode): pass
class StructDecl(AstNode): pass
class FuncDecl(AstNode): pass
class Transformation(AstNode): pass
class Attribute(AstNode): pass
class ErrorNode(AstNode): pass

class IfStmt(AstNode): members = ["cond", "block"]
class ElseStmt(AstNode): members = ["block"]
class WhileStmt(AstNode): members = ["cond", "block"]


class Token():
    __slots__ = ["type", "value", "file", "line", "col"]
    def __init__(self, type, value, file, line, col):
        self.type = type
        self.value = value
        self.file = file
        self.line = line
        self.col = col

    def __repr__(self):
        return "%s(%r) at %s:%s,%s" % (self.type, "" if self.value is None else self.value, self.file, self.line, self.col)
    def __eq__(self, rhs):
        return (self.type, self.value) == rhs


class TokenizerError(Exception):
    def __init__(self, msg, file, line, col, **kw):
        self.msg = msg
        self.file = file
        self.line = line
        self.col = col
        self.kw = kw

    def __str__(self):
        return "[%s:%s,%s] %s %s" % (self.file, self.line, self.col, self.msg, self.kw if self.kw else "")

class Tokenizer():
    double_char_ops = frozenset(["++", "--", ">>", "<<", "**", "&&", "||", ">=", "<=", "==", "!="])
    double_char_asgn = frozenset(["+=", "-=", "*=", "/=", "%=", "&=", "|=", ">>=", "<<="])
    triple_char_asgn = frozenset([">>=", "<<="])
    comment_markers = frozenset(["//", "/+", "/*"])
    keywords = frozenset(["if", "for", "else", "while", "true", "false", "null", "import", "module", "macro"])
    builtin_types = frozenset(["void", "bool", "char",
                     "uint8", "uint16", "uint32", "uint64", "uint128",
                     "sint8", "sint16", "sint32", "sint64", "sint128",
                     "float32", "float64", "float128"])

    def __init__(self, filename):
        self.filename = filename
        with open(filename, "r") as f:
            self.text = f.read()
        self.file_offset = 0
        self.col_num = 0
        self.line_num = 1

    def read_char(self):
        if self.file_offset >= len(self.text):
            return None
        ch = self.text[self.file_offset]
        self.col_num += 1
        if ch == "\n":
            self.line_num += 1
            self.col_num = 0
        self.file_offset += 1
        return ch

    def peek_char(self, offset=0):
        return self.text[self.file_offset+offset] if self.file_offset+offset < len(self.text) else None

    def fail(self, msg, **kw):
        raise TokenizerError(msg, self.filename, self.line_num, self.col_num, **kw)

    def read_string(self, mode, terminator):
        res = ""
        while True:
            ch = self.read_char()
            if ch is None:
                self.fail("EOF before seeing string terminator")

            if mode == "x":
                if ch == terminator:
                    break
                elif ch in "0123456789abcdefABCDEF":
                    res += ch
                else:
                    self.fail("Invalid character in hex string", char=ch)

            elif mode == "r":
                if ch == terminator:
                    break
                res += ch

            elif mode is None:
                if ch == "\\":
                    ch2 = self.peek_char()
                    if ch2 == "n":
                        self.read_char()
                        res += "\n"
                    elif ch2 == "r":
                        self.read_char()
                        res += "\r"
                    elif ch2 == "t":
                        self.read_char()
                        res += "\t"
                    elif ch2 == "\\":
                        self.read_char()
                        res += "\\"
                    elif ch2 == '"':
                        self.read_char()
                        res += '"'
                    elif ch2 == "x":
                        ch1 = self.read_char()
                        self.failIf(ch1 is None, "EOF while expecting hex char following \\x")
                        ch2 = self.read_char()
                        self.failIf(ch2 is None, "EOF while expecting hex char following \\x")
                        res += (ch1 + ch2).decode("hex")
                    elif ch2 is None:
                        self.fail("EOF while expecting character following '\\'")
                    else:
                        self.fail("Invalid escape sequence '\\%s'" % (ch,))
                elif ch == terminator:
                    break
                else:
                    res += ch

            else:
                assert False, mode
        if mode == "x":
            return res.decode("hex")
        else:
            return res

    def read_num(self, ch):
        ch2 = self.peek_char()
        if ch == "0" and ch2 in "xob":
            self.read_char()
            mode = ch2
            res = ""
        else:
            mode = "d"
            res = ch

        while True:
            ch = self.peek_char()
            if ch is None:
                break
            elif ch == "_":
                self.read_char()
                continue
            if mode == "b":
                if ch in "01":
                    res += self.read_char()
                elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ23456789":
                    self.fail("Invalid binary literal '%s'" % (ch,))
                else:
                    break
            elif mode == "o":
                if ch in "01234567":
                    res += self.read_char()
                elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ89":
                    self.fail("Invalid octal literal '%s'" % (ch,))
                else:
                    break
            elif mode == "x":
                if ch in "0123456789abcdefABCDEF":
                    res += self.read_char()
                elif ch in "ghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid hex literal '%s'" % (ch,))
                else:
                    break
            elif mode == "d":
                if ch in "0123456789":
                    res += self.read_char()
                elif ch == ".":
                    if self.peek_char(1) in "0123456789":
                        res += self.read_char()
                        mode = "frac"
                    else:
                        res += self.read_char()
                        res += "0"
                        mode = "frac"
                        break
                elif ch in "eE":
                    res += self.read_char()
                    mode = "exp1"
                elif ch in "abcdfghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid decimal literal '%s'" % (ch,))
                else:
                    break
            elif mode == "exp1":
                if ch in "+-":
                    res += self.read_char()
                    mode = "exp2"
                elif ch in "0123456789":
                    res += self.read_char()
                    mode = "exp3"
                elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid float exponent literal '%s'" % (ch,))
                else:
                    self.fail("Missing float exponent")
            elif mode == "exp2":
                if ch in "0123456789":
                    res += self.read_char()
                    mode = "exp3"
                elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid float exponent literal '%s'" % (ch,))
                else:
                    self.fail("Missing float exponent")
            elif mode == "exp3":
                if ch in "0123456789":
                    res += self.read_char()
                elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid float exponent literal '%s'" % (ch,))
                else:
                    break
            elif mode == "frac":
                if ch in "0123456789":
                    res += self.read_char()
                elif ch in "eE":
                    res += self.read_char()
                    mode = "exp1"
                elif ch in "abcdfghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ":
                    self.fail("Invalid decimal literal '%s'" % (ch,))
                else:
                    break
            else:
                assert False, mode

        if mode in ["exp3", "frac"]:
            return self.make_token("float", res)
        elif mode == "d":
            return self.make_token("int", int(res, 10))
        elif mode == "o":
            return self.make_token("int", int(res, 8))
        elif mode == "b":
            return self.make_token("int", int(res, 2))
        elif mode == "x":
            return self.make_token("int", int(res, 16))
        else:
            assert False, mode

    def read_comment(self, kind):
        nesting = 1
        while True:
            ch = self.read_char()
            if kind == "/":
                if ch == "\n" or ch is None:
                    break
            elif kind == "*":
                if ch is None:
                    self.fail("EOF in unclosed comment")
                elif ch == "*" and self.peek_char() == "/":
                    self.read_char()
                    break
            elif kind == "+":
                if ch is None:
                    self.fail("EOF in unclosed comment")
                elif ch == "/" and self.peek_char() == "+":
                    self.read_char()
                    nesting += 1
                elif ch == "+" and self.peek_char() == "/":
                    self.read_char()
                    nesting -= 1
                    if nesting == 0:
                        break
            else:
                assert False, kind

    def read_op(self, ch):
        tok = self.make_token("op", ch)
        ch2 = self.peek_char()
        ch3 = self.peek_char(1)
        double = None if ch2 is None else ch + ch2

        if double in self.comment_markers:
            self.read_char()
            self.read_comment(ch2)
            return None

        triple = None if ch3 is None or double is None else double + ch3

        if double in self.double_char_ops:
            self.read_char()
            tok.value = double
        elif double in self.double_char_asgn:
            self.read_char()
            tok.type = "asgn"
            tok.value = double
        elif triple in self.triple_char_asgn:
            self.read_char()
            self.read_char()
            tok.type = "asgn"
            tok.value = triple
        elif ch == "=":
            tok.type = "asgn"
            tok.value = ch

        return tok

    def read_identifier(self, ch):
        res = ch
        while True:
            ch = self.peek_char()
            if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789":
                self.read_char()
                res += ch
            else:
                break

        if res in self.keywords:
            return self.make_token("kwd", res)
        elif res in self.builtin_types:
            return self.make_token("blt", res)
        else:
            return self.make_token("id", res)

    def make_token(self, type, value=None):
        return Token(type, value, self.filename, self.line_num, self.col_num)

    def __iter__(self):
        while True:
            ch = self.read_char()
            if ch is None:
                break
            elif ch in " \t\r\n":
                continue
            elif ch in "xr" and self.peek_char() == '"':
                self.read_char()
                tok = self.make_token("str")
                tok.value = self.read_string(ch, '"')
                yield tok
            elif ch == '"':
                tok = self.make_token("str")
                tok.value = self.read_string(None, '"')
                yield tok
            elif ch in "{":
                yield self.make_token("begin")
            elif ch in "}":
                yield self.make_token("end")
            elif ch == ";":
                yield self.make_token("eos")
            elif ch == "@":
                yield self.make_token("at")
            elif ch in "()[]+-*~/%^&|~!=><:,":
                tok = self.read_op(ch)
                if tok:
                    yield tok
            elif ch in "0123456789":
                yield self.read_num(ch)
            elif ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
                yield self.read_identifier(ch)
            elif ch == ".":
                if self.peek_char() in "0123456789":
                    yield self.read_num("0.")
                else:
                    yield self.make_token("dot")
            else:
                self.fail("Invalid token '%s'" % (ch,))

class Parser():
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self._pos = 0
    def match(self, type, value=None):
        if self._pos >= len(self.tokens):
            return None
        tok = self.tokens[self._pos]
        if tok.type == type and (value is None or tok.value == value):
            self._pos += 1
            return tok
        else:
            return False
    def matchOrFail(self, type, value=None):
        t = self.match(type, value)
        if not t:
            self.fail("Expected %s(%s)" % (type, value))
        return t
    def fail(self, msg):
        raise Exception(msg)

    def pop_dotted_name(self):
        t = self.match("id")
        if not t:
            return None
        name = [t]
        while True:
            t = self.match("dot")
            if not t:
                break
            name.append(self.matchOrFail("id").value)
        return name

    def parse_module(self):
        self.matchOrFail("kwd", "module")
        name = self.pop_dotted_name()
        self.matchOrFail("eos")
        node = Module(name=name, children=[])
        while self._pos < len(self.tokens):
            node.children.append(self.parse_top_level())

    def parse_top_level(self):
        if self.match("kwd", "import"):
            return self.parse_import()
        elif self.match("kwd", "struct"):
            return self.parse_struct()
        else:
            self.fail("Invalid top-level element")

    def parse_import(self):
        #("kwd", "import").pop_dotted_name
        pass

#dotted_name = Seq(Id, Repeat(Seq(Dot, Id)), Eos)
#module_stmt = Seq(Kwd("module"), dotted_name, Eos)
#import_stmt = Seq(Kwd("import"), dotted_name, Opt(Seq(Op(":"), Id)), Eos)
block =
if_stmt = Seq(Kwd("if"), Op("("), expr, Op(")"), block)



if __name__ == "__main__":
    ast = Parser(Tokenizer("test.kr")).parse_module()
    print(ast)










