##pr4
from pprint import pprint
class Foo(object): # inheriting from object -- lowercase -- is required for "new style classes" in Python.
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __str__(self): # __str__() is what happens when you call str( instance_of_Foo)
        return "Foo(%s, %s)" % (self.a, self.b) # -- make sure you always reference fields as self.field_name or you will have errors.
    
f = Foo( "hi", "there")
print(f)

class Bar( Foo):
    def __init__(self, a, b, c, d):
        super( Bar, self).__init__(a, b)
        self.c = c
        self.d = d
    def __str__(self):
        return "Bar(%s, %s, %s, %s)" % (self.a, self.b, self.c, self.d)

b = Bar ("hi", "there", "from", "python")
print(b)

class LogicObject(object):
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return "{0}('{1}')".format(str(self.__class__)[17:-2], self.name)
    
    def __str__(self):
        return "{0}:'{1}'".format(str(self.__class__)[17:-2], self.name)
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name
    
    # Added after experimentation/errors.
    def __ne__(self, other):
        return not (self.name == other.name)
    
    def __hash__(self):
        return hash(self.name)

class Variable(LogicObject):
    def __init__(self, name):
        super(Variable, self).__init__(name)

class Constant(LogicObject):
    def __init__(self, name):
        super(Constant, self).__init__(name)

class Relation(Constant):
    def __init__(self, name):
        super(Relation, self).__init__(name)
        
x = Variable("x")
Fred = Constant("Fred")
George = Constant("Fred") # whoops...
FredRelation = Relation("Fred")
son = Relation("son")
print(x)
print(Fred)
print(FredRelation)

print("Is Fred the same as George? {0}".format(Fred == George))
print("Is Fred the same as FredRelation? {0}".format(Fred == FredRelation))

print("son != son? {0}".format(son != son))

class Expression(object):
    def __init__(self, relation, args=None):
        if not isinstance(relation, Relation):
            print("Need a relation to apply to the arguments...")
            return
        self.relation = relation
        if args is None:
            print("Need at least one argument to a Relation to form an expression...")
            return
        if not isinstance(args, list):
            args = [args]
        self.args = args
        for arg in self.args:
            if not isinstance(arg, (Constant, Variable)):
                print("All arguments after the Relation must be Constant or Variable")
                return

    def __repr__(self):
        string = "Expression({r}, ".format(r=self.relation)
        string += ", ".join(arg.__repr__() for arg in self.args) + ")"
        return string
        
    def __str__(self):
        string = "{r}?(".format(r=self.relation)
        string += ", ".join(str(arg) for arg in self.args) + ")"
        return string
    
expr = Expression(Relation("son"), [Constant("Fred"), Variable("x")])
print(expr)

def unify(expr1, expr2, verbose=False):
    if len(expr1.args) != len(expr2.args):  # Rule 1
        print("Can't very well unify 2 expressions when they don't have the same prototype...")
        return None
    if expr1.relation != expr2.relation:   # Rule 2
        print("Pretty sure unification only makes sense on the same Relation...")
        return None
    substitutions = {}
    while True:
        subs = False
        for idx, (arg1, arg2) in enumerate(zip(expr1.args, expr2.args)):
            if isinstance(arg1, Constant) and isinstance(arg2, Constant) and arg1 != arg2:   # Rule 3
                if verbose:
                    print("{0} and {1} are incompatible Constants. Fail".format(arg1, arg2))
                return None
            if isinstance(arg1, Constant) and isinstance(arg2, Variable):  # Rule 4
                if arg2 not in substitutions:
                    if verbose:
                        print("Substitutiong {0} for {1}".format(arg1, arg2))
                    substitutions[arg2] = arg1
                    subs = True
                elif substitutions[arg2] != arg1:
                    if verbose:
                        print("{0} has substitution {1} which doesn't match {2}".format(arg2, substitutions[arg2], arg1))
                    return None
                # else: substitution already matches constant, do nothing.
            elif isinstance(arg1, Variable) and isinstance(arg2, Constant):
                if arg1 not in substitutions:
                    if verbose:
                        print("Substitutiong {0} for {1}".format(arg2, arg1))
                    substitutions[arg1] = arg2
                    subs = True
                elif substitutions[arg1] != arg2:
                    if verbose:
                        print("{0} has substitution {1} which doesn't match {2}".format(arg1, substitutions[arg1], arg2))
                    return None
                # else: substitution already matches constant, do nothing.
        # if no new substitutions, then we're as done as can be.
        if subs == False:
            break
    return substitutions

def wrapperUnify(expr1, expr2, verbose=False):
    subs = unify(expr1, expr2, verbose)
    if subs is None:
        print("FAIL: {0} and {1} could NOT be unified\n".format(expr1, expr2))
    elif len(subs) == 0:
        print("EHH: {0} and {1} are trivially unified, in that no substitutions were needed\n".format(expr1, expr2))
    else:
        print("SUCCESS: {0} and {1} were unified, with the following substitution list:\n{2}\n".format(expr1, expr2, subs))
    return

relSon = Relation("son")
relParent = Relation("parent")
relSelf = Relation("self")
relYabba = Relation("yabba")
conFred, conWilma, conPebbles = Constant("Fred"), Constant("Wilma"), Constant("Pebbles")
conBarney, conBamBam = Constant("Barney"), Constant("BamBam")
varX, varY, varZ = Variable("x"), Variable("y"), Variable("z")

verbose = False
wrapperUnify(Expression(relSon, [conFred, varX]), Expression(relSon, [varY, conBamBam]), verbose)
wrapperUnify(Expression(relParent, [conPebbles, varX, varY]), Expression(relParent, [varZ, conFred, conWilma]), verbose)
wrapperUnify(Expression(relSelf, [varX, varX]), Expression(relSelf, [conFred, varX]), verbose)
wrapperUnify(Expression(relSelf, [varX, varX]), Expression(relSelf, [varX, conFred]), verbose)
wrapperUnify(Expression(relYabba, conFred), Expression(relYabba, [varY]), verbose)

wrapperUnify(Expression(relParent, [conPebbles, varX]), Expression(relParent, [varZ, conFred, conWilma]), verbose)
wrapperUnify(Expression(relSon, [varX, conFred]), Expression(relParent, [conBamBam, varY]), verbose)
wrapperUnify(Expression(relSon, [conFred, varX]), Expression(relSon, [conBamBam, varY]), verbose)
wrapperUnify(Expression(relSelf, [varX, conFred]), Expression(relSelf, [conBarney, varX]), verbose)
wrapperUnify(Expression(relSelf, [varX, varX]), Expression(relSelf, [conFred, conWilma]), verbose)



