class Token():
    __slots__ = ["type", "value", "file", "line", "col", "idx"]
    def __init__(self, type, value, file, line, col):
        self.type = type
        self.value = value
        self.file = file
        self.line = line
        self.col = col
        self.idx = None

    def __repr__(self):
        #return "%s(%r) at %s:%s,%s" % (self.type, "" if self.value is None else self.value, self.file, self.line, self.col)
        return "%s(%r)" % (self.type, "" if self.value is None else self.value)
    def __eq__(self, rhs):
        return (self.type, self.value) == rhs


class BadSyntax(Exception):
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
        raise BadSyntax(msg, self.filename, self.line_num, self.col_num, **kw)

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
            return self.make_token("flt", res)
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
            if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789.":
                self.read_char()
                res += ch
            else:
                break

        return self.make_token("dotted" if "." in res else "sym", res)

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

    def get_stream(self):
        tokens = []
        for i, t in enumerate(self):
            t.idx = i
            tokens.append(t)
        return TokenStream(tokens)

class TokenStream():
    def __init__(self, tokens):
        self._tokens = tokens
        self._pos = 0
    def tell(self):
        return self._pos
    def seek(self, pos):
        assert (pos <= self._pos)
        self._pos = pos
    def peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None
    def pop(self):
        if self._pos >= len(self._tokens):
            return None
        t = self._tokens[self._pos]
        self._pos += 1
        return t
    def match(self, type, value=NotImplemented):
        if self._pos >= len(self._tokens):
            return False
        t = self._tokens[self._pos]
        if t.type == type and (value is NotImplemented or value == t.value):
            self._pos += 1
            return t
        else:
            return False

#=======================================================================================================================

class Production():
    __slots__ = ("keep")
    def parse(self, tokens):
        raise NotImplementedError()
    def __add__(self, rhs):
        if isinstance(self, Seq):
            if isinstance(rhs, Seq):
                self.prods.extend(rhs.prods)
            else:
                self.prods.append(rhs)
            return self
        elif isinstance(rhs, Seq):
            rhs.prods.insert(0, self)
            return rhs
        else:
            return Seq(self, rhs)
    def __invert__(self):
        return Opt(self)
    def __rshift__(self, func):
        return Transform(self, func)
    def __or__(self, rhs):
        if isinstance(self, FirstMatch):
            if isinstance(rhs, FirstMatch):
                self.prods.extend(rhs.prods)
            else:
                self.prods.append(rhs)
            return self
        elif isinstance(rhs, FirstMatch):
            rhs.prods.insert(0, self)
            return rhs
        else:
            return FirstMatch(self, rhs)

class Seq(Production):
    __slots__ = ("prods",)
    def __init__(self, *prods):
        self.prods = list(prods)
    def parse(self, tokens):
        start_pos = tokens.tell()
        res = []
        for p in self.prods:
            r = p.parse(tokens)
            if r is False:
                tokens.seek(start_pos)
                return False
            if getattr(p, "keep", True):
                res.append(r)
        end_pos = tokens.tell()
        return res
    def __repr__(self):
        return "Seq(%s)" % (", ".join(repr(p) for p in self.prods),)

class Opt(Production):
    __slots__ = ("prod",)
    def __init__(self, prod):
        self.prod = prod
    def parse(self, tokens):
        r = self.prod.parse(tokens)
        return None if r is False else r
    def __repr__(self):
        return "Opt(%r)" % (self.prod,)

class Tok(Production):
    __slots__ = ("type", "value")
    def __init__(self, type, value=NotImplemented, keep=False):
        self.type = type
        self.value = value
        self.keep = keep
    def parse(self, tokens):
        return tokens.match(self.type, self.value)
    def __repr__(self):
        return "%s(%r)" % (self.type, "" if self.value is NotImplemented else self.value)

class Repeat(Production):
    __slots__ = ("prod",)
    def __init__(self, prod):
        self.prod = prod
    def parse(self, tokens):
        res = []
        while True:
            r = self.prod.parse(tokens)
            if r is False:
                break
            res.append(r)
        return res
    def __repr__(self):
        return "Repeat(%r)" % (self.prod,)

class Transform(Production):
    __slots__ = ("prod", "func")
    def __init__(self, prod, func):
        self.prod = prod
        self.func = func
    def parse(self, tokens):
        p = self.prod.parse(tokens)
        if p is False:
            return False
        return self.func(p)
    def __repr__(self):
        return "Transform(%r, %r)" % (self.prod, self.func)

class Lazy(Production):
    __slots__ = ("prodfunc",)
    def __init__(self, prodfunc):
        self.prodfunc = prodfunc
    def parse(self, tokens):
        return self.prodfunc().parse(tokens)
    def __repr__(self):
        return "Lazy(lambda: %r)" % (self.prodfunc(),)

class FirstMatch(Production):
    __slots__ = ("prods",)
    def __init__(self, *prods):
        self.prods = list(prods)
    def parse(self, tokens):
        pos = tokens.tell()
        for p in self.prods:
            r = p.parse(tokens)
            if r is not False:
                return r
            tokens.seek(pos)
        return False
    def __repr__(self):
        return "FirstMatch(%s)" % (", ".join(repr(p) for p in self.prods),)

#=======================================================================================================================

class AstNode():
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join("%s=%r" % (k, v) for k, v in self.__dict__.items()))

    def show(self, nesting=0):
        attrs = ", ".join("%s=%r" % (k, v) for k, v in self.__dict__.items() if k != "body")
        print("%s%s(%s)" % ("  " * nesting, self.__class__.__name__, attrs))
        body = getattr(self, "body", ())
        if hasattr(body, "show"):
            body.show(nesting+1)
        elif hasattr(body, "__iter__"):
            for child in body:
                if hasattr(child, "show"):
                    child.show(nesting+1)
                else:
                    print("%s%r" % ("  " * (nesting+1), child))
        else:
            print("%s%r" % ("  " * (nesting+1), body))


class Module(AstNode): pass
class Import(AstNode): pass
class VarDef(AstNode): pass
class VarRef(AstNode): pass

class StructDef(AstNode): pass
class TypeName(AstNode): pass
class FuncDef(AstNode): pass
class If(AstNode): pass
class Else(AstNode): pass
class While(AstNode): pass
class Return(AstNode): pass
class Continue(AstNode): pass
class Break(AstNode): pass


eos = Tok("eos")
comma = Tok("op", ",")
colon = Tok("op", ":")
begin = Tok("begin")
end = Tok("end")
at = Tok("at")

def kwd(n, keep=False): return Tok("sym", n, keep=keep)
def op(n, keep=True): return Tok("op", n, keep=keep)

word = Tok("sym", NotImplemented, keep=True) >> (lambda p: p.value)
dotted = (Tok("sym", NotImplemented, keep=True) | Tok("dotted", NotImplemented, keep=True)) >> (lambda p: p.value)
module_stmt = kwd("module") + dotted + eos >> (lambda p: Module(name=p[0]))
import_stmt = kwd("import") + dotted + ~(colon + word + Repeat(comma + word) >> (lambda prods: [prods[0]] + sum(prods[1], []))) + eos  >> (lambda p: Import(module=p[0], names=p[1]))
literal = Tok("str") | Tok("int") | Tok("flt")
expr = literal

macro = at + dotted >> (lambda p: p[0]) #+ ~(op("(", keep=False) +  + op(")", keep=False))
macros = Repeat(macro)

subscript = op("[", keep=False) + expr + op("]", keep=False) >> (lambda prods: prods[0])
typename = dotted + Repeat(subscript) >> (lambda prods: TypeName(name=prods[0], subscripts=prods[1]))
var_def = macros + typename + word + ~(Tok("asgn", "=") + expr) + eos >> (lambda p: VarDef(macros=p[0], type=p[1], name=p[2], init=p[3]))
struct_body = Repeat(var_def)
struct_def = macros + kwd("struct") + word + begin + struct_body + end >> (lambda p: StructDef(macros=p[0], name=p[1], body=p[2]))
func_arg = macros + typename + word >> (lambda p: VarDef(macros=p[0], type=p[1], name=p[2]))
func_args = Repeat(func_arg + comma) + Opt(func_arg) >> (lambda prods: (prods[0] + [prods[1]]) if prods[1] else prods[0])

suite = (begin + Repeat(Lazy(lambda: stmt)) + end) | Lazy(lambda: stmt)
else_stmt = kwd("else") + suite >> (lambda p: Else(body=p[0]))
if_stmt = kwd("if") + op("(", keep=False) + expr + op(")", keep=False) + suite + ~else_stmt >> (lambda p: If(cond=p[0], body=p[1], else_=p[2]))
while_stmt = kwd("while") + op("(", keep=False) + expr + op(")", keep=False) + suite >> (lambda p: While(cond=p[0], body=p[1]))
return_stmt = kwd("return") + ~expr + eos >> (lambda p: Return(expr=p[0]))
break_stmt = kwd("break") + eos >> (lambda p: Break(label=None))
continue_stmt = kwd("continue") + eos >> (lambda p: Continue(label=None))
#macro_stmt = macro + suite

stmt = var_def | if_stmt | while_stmt | return_stmt | break_stmt | continue_stmt #| macro_stmt
func_def = macros + typename + word + op("(", keep=False) + func_args + op(")", keep=False) + begin + Repeat(stmt) + end >> (lambda p: FuncDef(macros=p[0], type=p[1], name=p[2], args=p[3], body=p[4]))


top_level = import_stmt | struct_def | func_def
def mk_module(p):
    p[0].body = p[1]
    return p[0]
root = module_stmt + Repeat(top_level) >> mk_module

if __name__ == "__main__":
    tokens = Tokenizer("test.kr").get_stream()
    root.parse(tokens).show()
    assert tokens.pop() is None








