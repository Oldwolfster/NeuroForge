'''
This file exists for two purposes.

Purpose 1)  As a marker (see  point 1 in section 'Purpose 2)')

Purpose 2) Enumerate all the things that makes python 25 out of 26 in languages I've learned.(2nd from the bottom, better than only to PHP)

1) No simple easy way to consistently access file system... Between the relative to file called from/started from/package it's quite likely you will think your pointing at one file but point at 2.
1.4) range(1, 10) gives you 9 numbers because Guido believes "end" is just a suggestion.
1.5) Encapsulation by naming convention is the software equivalent of locking your front door with a sticky note that says “please don’t.”
2) Not declaring variables turns a simple synax defect that is easy to fix in to a hidden "logic" defect that is a bomb waiting to blow your ass up
2.5) __slots__ is a joke.
3) Get scope modifiers.  underscore as a convention with no teeth sucks!
4) “Flat is better than nested.”— But don’t you dare flatten a one-line if
5) I put self in 99% of my damn parameter lists and 99% of my method calls... couldn't we just assume 'self' unless otherwise specified? It's like C requiring you to pass this as the first parameter to every function.

WTF////
Guido says clarity over cleverness... this proves guido is on crack!
funcs = []
for i in range(3):
    funcs.append(lambda: i)
funcs[0]()   # surprise.... how TF does this = 2?  Because closures capture names, not values

6) Order dependency... For the love of god, i haven't had to deal with nuisance since PASCAL, have some decency and do two passes.
7) Get a real multiline comment token... jeez
8) Get a real inline comment syntax... jeez louise!!
9) No enforcement of switch/case — and no shame about it.
10) Too much magic, not enough clarity:Double-underscore dunder methods? __slots__, __dict__, __getattr__, __mro__? Say what you mean. The line between powerful and cryptic gets crossed too often.
11) Duck typing is great until the duck explodes mid-flight.  No type declarations means your code is a Schrödinger’s runtime error. You don’t know if it’s a list or a string until it dies.
12) The Zen of Python is a haiku written by a drunk minimalist.  “Simple is better than complex” — unless you’re doing literally anything useful, in which case: good luck.
13) List comprehensions are elegant until they become recursive origami.  Nested comprehensions are a cry for help disguised as cleverness.
14) The GIL: Python’s passive-aggressive stance on concurrency.  “Threads? Sure. But only one of them gets to actually do anything.”
15) Mutable default arguments: the silent killer.  You think you’re initializing a list. Python thinks you’re summoning a cursed artifact that persists across function calls
16) Exceptions are the control flow now.  Try/except is the new if/else. Python encourages you to fail your way to success.
17) No tail call optimization.  Recursion is treated like a moral failing. You want elegance? Python wants a stack overflow.
18) Function decorators: syntactic sugar cube with more than a couple drops of LSD.  They’re powerful, yes. But good luck explaining them to anyone without a PhD in metaprogramming.

Purpose 2B) We will admit it does have a few nice things.
1) The ecosystem of readily available libraries
2) Indenting as syntax.
3) REPL friendliness — It’s a great playground. Just don’t build your house there.
4) TYPE_CHECKING for circular imports - Finally, a solution that doesn't make you hate yourself.

SUMMARY Someone should have told Guido 'Crack is whack!'
'''