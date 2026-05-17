# ML Patterns — In Context

A living document. Updated after every video, paper, or project.
Each pattern is anchored to the exact moment and problem that produced it.
The goal: recall through context, not through abstract definitions.

---

## How to Use This File

- **When stuck** — find the problem that looks like yours, re-read the context
- **Before a new project** — scan for patterns that might apply
- **After a video** — add new patterns with timestamp and problem that triggered them

---

## How to Add New Patterns

```
### Pattern N — Name
**Video/Source:** name + timestamp
**The problem at that moment:** what specific thing wasn't working or wasn't clear
**What was tried or observed:** what led to the insight
**The pattern:** the reusable takeaway
**Key insight:** the one sentence worth remembering
```

---

## Video 1 — Karpathy "The spelled-out intro to neural networks and backpropagation" (micrograd)

---

### Pattern 0a — Why Wrap Numbers in a Value Object
**Video:** micrograd video, ~10 mins in
**The problem at that moment:** Karpathy wanted to do math on numbers but also track how those numbers were computed so he could later work out gradients. A plain Python float has no memory of where it came from.
**What was observed:** wrapping a number in a `Value` object lets you attach extra information to it:
```python
class Value:
    def __init__(self, data):
        self.data = data      # the actual number
        self.grad = 0.0       # gradient — how much loss changes if this changes
        self._prev = set()    # which Values produced this one
        self._op = ''         # which operation produced this (+, *, tanh...)
        self._backward = lambda: None  # how to compute gradients for this op
```
Every time you do `a + b`, a new `Value` is created that remembers `a` and `b` as its parents and `+` as its operation. This builds a graph of the entire computation.
**Key insight:** The `Value` class is a number with a memory. It remembers every operation that produced it. That memory is what makes backprop possible — you need to know how every number was computed before you can work out how changing it affects the loss.

---

### Pattern 0b — The Computation Graph
**Video:** micrograd video, ~15 mins in
**The problem at that moment:** after building a few `Value` operations, Karpathy visualised the result with `draw_dot()`. The picture showed nodes (values) connected by edges (operations).
**What was observed:** every math operation creates a new node and connects it to its inputs:
```
a = Value(2.0)    b = Value(-3.0)    c = Value(10.0)
         \              /
          \            /
           d = a * b = -6.0
                \
                 \
                  L = d + c = 4.0
```
Forward pass flows left to right — data flows through operations producing the final output. Backward pass flows right to left — gradients flow back through the same graph.
**Key insight:** The computation graph is built automatically just by doing math with `Value` objects. You never explicitly construct it. `_prev` on each node records its parents, forming the graph implicitly as you compute.

---

### Pattern 0b.5 — The Chain Rule: How Gradients Flow Through a Graph
**Video:** micrograd video, ~20 mins in
**The problem at that moment:** the network is a chain of operations — `a → multiply → d → add → L`. When you want to know "how does changing `a` affect `L`?", you can't just look at `a` and `L` directly because there are operations in between.
**What was observed:** the chain rule says — to find the gradient of the final output with respect to any input, multiply the local gradients along every path connecting them:
```
L = d + c
d = a * b

How much does L change when a changes?

dL/da = dL/dd × dd/da
      = 1.0   × b.data
      = 1.0   × -3.0
      = -3.0
```
In English: "L changes by 1 for every unit change in d (that's dL/dd). d changes by b (-3.0) for every unit change in a (that's dd/da). So L changes by 1 × -3.0 = -3.0 for every unit change in a."

In the code, this is why every `_backward` function multiplies by `out.grad`:
```python
# multiplication backward — chain rule in action
def _backward():
    self.grad += other.data * out.grad   # dd/da × dL/dd
    other.grad += self.data * out.grad   # dd/db × dL/dd
                             ↑
                    this is the chain rule — multiply local gradient
                    by the gradient already flowing back from the output
```
**Key insight:** Each node only needs to know its own local derivative (how its output changes when its input changes). The chain rule connects them — multiply local gradients along the path from output back to any input. This is the entire mathematical foundation of backprop. Every `_backward` function in `value.py` is just the chain rule applied to one specific operation.

---

### Pattern 0c — Each Operation Defines Its Own Backward
**Video:** micrograd video, ~25 mins in
**The problem at that moment:** how does backprop know the gradient rule for addition vs multiplication vs tanh? Each operation has a different derivative.
**What was observed:** each operation in `value.py` defines its own `_backward` closure — the local gradient rule for that specific operation:
```python
# addition: gradient passes through unchanged to both inputs
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad += 1.0 * out.grad   # gradient of addition = 1
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

# multiplication: each input's gradient = other input's value × output gradient
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out
```
**Key insight:** Each operation only needs to know its own local derivative rule. It doesn't need to know anything about the rest of the network. This is the chain rule in practice — each node computes its local gradient and passes it back to its parents. The whole network's gradients emerge from each node doing its small local job.

---

### Pattern 0c.1 — Local Derivative Formulas for Each Operation (The Math)
**Video:** micrograd video, ~25 mins in
**The problem at that moment:** where do the numbers inside each `_backward` come from? Why is the gradient of addition 1? Why does multiplication swap the inputs?
**What was observed:** these come directly from calculus derivatives — but you only need to memorise 4:
```
Addition:       d/da (a + b) = 1        → gradient passes through unchanged
                d/db (a + b) = 1        → same for the other input

Multiplication: d/da (a * b) = b        → gradient = the OTHER value
                d/db (a * b) = a        → gradient = the OTHER value

Power:          d/da (a^n)   = n * a^(n-1)   → bring exponent down, reduce power by 1

tanh:           d/da tanh(a) = 1 - tanh²(a)  → 1 minus the output squared
```
In code:
```python
# addition — why 1.0 * out.grad
self.grad += 1.0 * out.grad       # d(a+b)/da = 1, times chain rule

# multiplication — why other.data and self.data
self.grad += other.data * out.grad   # d(a*b)/da = b
other.grad += self.data * out.grad   # d(a*b)/db = a

# tanh — why (1 - t**2)
self.grad += (1 - t**2) * out.grad   # d(tanh)/da = 1 - tanh²(a)

# power — why other * self.data**(other-1)
self.grad += other * (self.data**(other-1)) * out.grad  # standard power rule
```
**Key insight:** You don't need to derive these — just memorise the 4 rules above. Every `_backward` in `value.py` is just one of these rules multiplied by `out.grad` (the chain rule part).

---

### Pattern 0c.2 — Why `_backward` is a Closure (Not a Regular Function)
**Video:** micrograd video, ~25 mins in
**The problem at that moment:** `_backward` is defined *inside* `__mul__`, `__add__` etc. Why does it need to be defined there and not outside?
**What was observed:** `_backward` needs access to `self`, `other`, and `out` — the specific values that existed at the moment the operation was performed. By defining it inside the operation, it captures those values permanently:
```python
def __mul__(self, other):
    out = Value(self.data * other.data, ...)
    def _backward():
        self.grad += other.data * out.grad  # ← other and out are captured here
                                            #   they stay alive even after __mul__ returns
    out._backward = _backward
    return out
```
If `_backward` were defined outside, it would have no way to know which `self`, `other`, and `out` it belongs to.
**Key insight:** Closures let `_backward` remember the specific values it was created with. Each operation creates its own `_backward` that is permanently attached to the exact inputs and output of that operation. This is what makes the computation graph work — each edge in the graph carries its own gradient function.



---

### Pattern 0d — += Not = When Accumulating Gradients
**Video:** micrograd video, ~27 mins in
**The problem at that moment:** in the `_backward` functions, gradients are written as `self.grad += ...` not `self.grad = ...`. Why accumulate instead of assign?
**What was observed:** a single `Value` can be used in multiple places in the computation graph. For example if `a` appears in both `a*b` and `a+c`, then `a.grad` receives gradient contributions from both paths. Assigning `=` would overwrite the first contribution — you'd lose it.
```python
a = Value(2.0)
d = a * b    # a gets gradient from here
e = a + c    # a also gets gradient from here
L = d + e
L.backward() # a.grad must be sum of both contributions
```
**Key insight:** `+=` because a node can have multiple children in the graph, each contributing a gradient. The total gradient is the sum of all contributions — this is the multivariate chain rule. Using `=` instead of `+=` silently drops gradient contributions and gives wrong weight updates.

---

### Pattern 0d.5 — Why loss.grad Starts at 1.0
**Video:** micrograd video, ~35 mins in
**The problem at that moment:** `backward()` sets `self.grad = 1.0` before starting. Why 1.0 and not something else?
**What was observed:** the loss is the final output. The gradient of anything with respect to itself is always 1 — "if I change the loss by 1, the loss changes by 1." This is the mathematical starting point for the chain rule to work backwards from:
```python
self.grad = 1.0   # dL/dL = 1  ← always true, this is where backprop starts
for node in reversed(topo):
    node._backward()  # chain rule flows backwards from here
```
Without setting it to 1.0, every gradient would be multiplied by 0 (the default) and nothing would update.
**Key insight:** `dL/dL = 1` is always mathematically true. It is the seed that starts the chain rule flowing. Every other gradient in the network is computed relative to this starting point.

---

### Pattern 0e — Topological Sort Before Backward
**Video:** micrograd video, ~35 mins in
**The problem at that moment:** `loss.backward()` needs to visit every node and call its `_backward`. But in what order? You must process a node only after all its children have already computed their gradients — otherwise you're passing incomplete gradients backwards.
**What was observed:** Karpathy built a topological sort — an ordering where every node appears after all the nodes it depends on:
```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)  # children first
            topo.append(v)        # then this node

    build_topo(self)
    self.grad = 1.0               # loss gradient = 1 (starting point)
    for node in reversed(topo):   # process in reverse = output → inputs
        node._backward()
```
**Key insight:** Backprop must go from output to inputs — reversed topological order. The loss node starts with gradient 1.0 ("the loss changes by 1 for every 1 unit change in itself"). Gradients then flow backward through the graph, each node computing how much it contributed to the loss.

---

### Pattern 0f — __radd__ for Compatibility With Python Built-ins
**Video:** micrograd video / discovered during nn.py training loop
**The problem at that moment:** the training loop used Python's built-in `sum()` to add up individual losses: `loss = sum((pred - y) ** 2 ...)`. This failed with `TypeError: unsupported operand type(s) for +: 'int' and 'Value'` because Python's `sum()` starts from integer `0` and tries `0 + Value(...)`.
**What was observed:** Python tries `int.__add__(Value)` first — which fails because int doesn't know about Value. Then Python tries `Value.__radd__(int)` as a fallback — the "right add". Adding `__radd__` to `Value` catches this:
```python
def __radd__(self, other):   # called when: int + Value
    return self + other      # just flip it — same as Value + int
```
**Key insight:** Whenever you write a custom numeric class in Python, add `__radd__`, `__rmul__` etc. alongside the regular operators. Without them, your class breaks silently when used with Python built-ins like `sum()`. This pattern appears any time you wrap numbers in custom objects.

---

---

### Pattern 0g — Why Weights Must Be Random Not Equal
**Video:** micrograd video, weight initialisation
**The problem at that moment:** why initialise weights randomly? Why not start them all at 0 or all at the same value?
**What was observed:** if all weights start equal, every neuron in a layer computes the exact same thing and gets the exact same gradient. They all update identically forever — effectively you have 1 neuron pretending to be many:
```
weights all = 0.5:
  neuron 1: output = 0.5×x1 + 0.5×x2 = 1.5  grad = 0.3
  neuron 2: output = 0.5×x1 + 0.5×x2 = 1.5  grad = 0.3  ← identical, learns nothing new
  neuron 3: output = 0.5×x1 + 0.5×x2 = 1.5  grad = 0.3  ← identical
```
Random initialisation breaks this symmetry — each neuron starts different, gets different gradients, and learns to detect different features.
**Key insight:** Equal initialisation causes "symmetry" — all neurons are identical and stay identical. Random initialisation breaks symmetry so each neuron can specialise. In micrograd: `Value(random.uniform(-1, 1))`. In PyTorch this happens automatically.

---

### Pattern 0h — __pow__ Must Be Added for the ** Operator
**Video:** micrograd video / discovered during nn.py training loop
**The problem at that moment:** the training loop used `(pred - y) ** 2` to compute squared error loss. This failed with `TypeError: unsupported operand type(s) for ** or pow(): 'Value' and 'int'` because `Value` had a `pow` method but Python's `**` operator looks for `__pow__`.
**What was observed:** Python's operators map to dunder methods. `**` calls `__pow__`, not `pow`. Adding `__pow__ = pow` wires them together:
```python
def pow(self, other):          # the implementation
    out = Value(self.data ** other, ...)
    ...

__pow__ = pow                  # makes ** operator call this method
```
Also: the exponent must stay a plain number (not a `Value`) because the gradient formula `other * self.data**(other-1)` requires plain arithmetic on `other`.
**Key insight:** Python operators (`+`, `*`, `**`) map to dunder methods (`__add__`, `__mul__`, `__pow__`). If you implement the method but not the dunder, the operator silently fails. Always add both when building a custom numeric class.

---

### Pattern 1 — Why Activation Functions Exist
**Video:** micrograd video, ~30 mins in
**The problem at that moment:** Karpathy had stacked two linear operations (weight × input + bias) and asked — what's the point of having two layers if they just collapse into one?
**What was observed:** mathematically, `W2(W1x + b1) + b2` simplifies to a single `Wx + b`. Two layers of pure linear math equals one layer. Depth is meaningless without something non-linear in between.
**The pattern:**
```
linear → linear → linear  =  still just linear (useless depth)
linear → tanh → linear    =  genuinely deeper (useful depth)
```

**Why one S-curve is not enough — but many S-curves can make any shape:**

A single neuron with tanh produces one S-shaped curve. That alone is limited. But a layer of neurons each produces its own S-curve — shifted, stretched, or flipped differently by its own weights and bias:
```
neuron 1: S-curve shifted left,  steep         (bias pulls it left, large weights)
neuron 2: S-curve shifted right, gentle        (bias pushes right, small weights)
neuron 3: S-curve flipped upside down          (negative weights flip it)
neuron 4: S-curve shifted far right, very flat (large bias, tiny weights)
```
The next layer then combines all of these with its own weights:
```
0.8×(neuron1) + (-0.3)×(neuron2) + 0.5×(neuron3) + ...
= some complex shape that looks nothing like an S-curve
```
Think of it like mixing paint — you only have red, blue, yellow but by mixing different amounts you can make any colour. tanh is your basic colour. Each neuron is a differently scaled version. The next layer mixes them into any shape the data requires.

**This is not intuition — it is a proven theorem:**

The **Universal Approximation Theorem** states:
> A neural network with even one hidden layer and enough neurons can approximate any continuous function to any desired accuracy.

It doesn't matter what shape the problem is — circular boundaries, spirals, jagged edges. Given enough neurons the network can learn it.

**Why depth is more efficient than just making one wide layer:**
```
Shallow (1 layer, many neurons):  tries to learn everything at once
Deep (many layers, fewer neurons):
    layer 1: learns simple features  (edges, basic patterns)
    layer 2: combines them           (shapes, higher patterns)
    layer 3: combines those          (objects, complex concepts)
```
Each layer builds on what the previous learned. Far more efficient than one giant layer.

**Key insight:** One neuron = one S-curve. Many neurons = many S-curves each shifted/stretched/flipped differently by weights and bias. Next layer combines them. The Universal Approximation Theorem proves that with enough neurons and layers, any shape can be approximated. The activation function is what makes this possible — without it everything collapses to a straight line regardless of how many layers you have.

---

### Pattern 2 — Why Bias Exists
**Video:** micrograd video, neuron implementation
**The problem at that moment:** building the Neuron class — why add a separate bias term instead of just using the weights?
**What was observed:** if all inputs are 0, then `w1×0 + w2×0 + ... = 0` regardless of weights. tanh(0) = 0. The neuron is stuck at 0 and the weights cannot fix it.
**The pattern:**
```
output = tanh(w1×x1 + w2×x2 + b)
                              ↑
                    bias shifts the whole sum
                    independently of inputs
```
**Key insight:** Bias lets the neuron fire (or not fire) independent of input values. It's the "baseline excitement" of the neuron. Without it, the neuron cannot activate when all inputs are zero, no matter what the weights are.

---

### Pattern 3 — Gradient Points Uphill, So Go Opposite
**Video:** micrograd video, gradient descent section
**The problem at that moment:** after computing gradients via backward(), which direction do you move the weights?
**What was observed:** gradient of a weight tells you "if I increase this weight, loss goes up by this much." We want loss to go down, so we move in the opposite direction.
**The pattern:**
```
weight = weight - learning_rate × gradient
                ↑
            minus sign = opposite direction = downhill
```
**Key insight:** Gradient descent is literally just: the gradient points uphill, so subtract it to go downhill. The learning rate controls step size — too large and you overshoot the minimum, too small and training takes forever.

---

### Pattern 4 — Zero Gradients Before Every Backward Pass
**Video:** micrograd video, training loop
**The problem at that moment:** training loop ran and loss was going in the wrong direction. Root cause: gradients from the previous step were still sitting on the weights and accumulating.
**What was observed:**
```
step 1: grad = 2.0  → update weight by -0.02
step 2: grad = 2.0 again, but old grad still there → total = 4.0 → update by -0.04  ← wrong
```
**The pattern:**
```
# always in this order:
for p in model.parameters():
    p.grad = 0.0      # ← zero first
loss.backward()       # ← then compute fresh gradients
```
**Key insight:** Gradients accumulate by default in PyTorch. If you don't zero them, each step corrupts the next. This is a silent bug — no error, wrong training.

---

### Pattern 5 — The Complete Training Loop Never Changes
**Video:** micrograd video, putting it all together
**The problem at that moment:** how do all the pieces (forward pass, loss, backward, update) connect into a working system?
**What was observed:** Karpathy assembled them in a specific order that is the same for every neural network ever trained — only the model and loss function change, never the loop structure.
**The pattern:**
```
for step in range(n_steps):
    predictions = model(x)          # 1. forward pass
    loss = compute_loss(predictions, y)  # 2. measure wrongness
    for p in model.parameters():
        p.grad = 0.0                # 3. zero gradients
    loss.backward()                 # 4. compute gradients
    for p in model.parameters():
        p.data -= lr * p.grad       # 5. update weights
```
**Key insight:** This exact structure appears in GPT training, robot policy training, image classifiers — everything. Internalise this loop. It never changes.

---

## Video 2 — Karpathy "The spelled-out intro to language modeling" (makemore / bigram)

---

### Pattern 5.5 — zip(w, w[1:]) for Sliding Window Over Consecutive Pairs
**Video:** makemore video, ~18 mins in
**The problem at that moment:** for every name, you need every consecutive pair of characters — (.→e), (e→m), (m→m) etc. How do you iterate over pairs without a clumsy index loop?
**What was observed:** `zip(w, w[1:])` pairs each element with the next one by zipping the sequence with itself offset by 1:
```python
w = ".emma."
list(zip(w, w[1:]))
# [('.','e'), ('e','m'), ('m','m'), ('m','a'), ('a','.')]
```
Compare to the clumsy version:
```python
for i in range(len(w) - 1):   # harder to read, same result
    ch1, ch2 = w[i], w[i+1]
```
**Key insight:** `zip(seq, seq[1:])` is the standard Python idiom for iterating consecutive pairs. It appears constantly in sequence processing — bigrams, sliding windows, comparing adjacent elements. Memorise it.

---

### Pattern 6 — 2D Lookup Table vs Dictionary for Bigrams
**Video:** makemore video, ~15 mins in
**The problem at that moment:** the natural first approach was a dict of (char1, char2) → count. But when you want to ask "given I just saw 'e', what is the probability of every possible next character?", you have to scan every key looking for ones starting with 'e'.
**What was observed:** a 2D array organises the same data so that one character = one row. "What comes after 'e'?" becomes a single row lookup instead of a full scan.
**The pattern:**
```
dict approach:   {('e','m'): 464, ('e','a'): 132, ...}  → scan to find all 'e' rows
2D array:        N[stoi['e']]  → entire row for 'e' instantly
```
**Key insight:** The 2D array isn't just faster — it's the same shape as the weight matrix the neural net version will use. The counting table and the neural net are solving the same problem with the same data structure.

---

### Pattern 7 — stoi and itos: Translating Between Characters and Numbers
**Video:** makemore video, ~20 mins in
**The problem at that moment:** the 2D count table uses integer indices but the data is characters. Need to convert in both directions — into the table and back out when generating names.
**What was observed:** two simple dictionaries solve this completely:
```python
stoi = {'a': 0, 'b': 1, ..., 'z': 25, '.': 26}  # string to int — going IN
itos = {0: 'a', 1: 'b', ..., 25: 'z', 26: '.'}  # int to string — going OUT
```
**The pattern:**
```
'e' → stoi → 5 → [model] → 13 → itos → 'm'
```
**Key insight:** The model only ever sees numbers. stoi/itos are the translation layer at the boundary. Every ML system that deals with non-numerical data (text, categories, actions) needs this same boundary translation.

---

### Pattern 7.5 — .float() Before Division on Integer Tensors
**Video:** makemore video, ~30 mins in
**The problem at that moment:** the count matrix `N` was created with `dtype=torch.int32`. Dividing an integer tensor by another integer tensor gives wrong results (integer division, not float division).
**What was observed:**
```python
N = torch.zeros((27, 27), dtype=torch.int32)  # integer counts
P = N / N.sum(...)   # wrong — integer division truncates decimals

P = N.float()        # convert to float first
P /= P.sum(...)      # correct — proper decimal division
```
**Key insight:** PyTorch tensors preserve their dtype through operations. Integer tensor / integer = integer (decimals truncated). Always call `.float()` before any division that should produce decimal probabilities. This is a silent bug — no error, wrong numbers.

---

### Pattern 7.6 — Smoothing: Add 1 to Counts to Avoid log(0)
**Video:** makemore video, loss calculation section
**The problem at that moment:** some bigrams never appear in the training data — their count is 0 and their probability is 0. When computing loss with `-log(probability)`, `log(0) = -infinity`. The loss becomes infinite and training breaks.
**What was observed:** adding 1 to every count before computing probabilities ensures no probability is ever exactly 0:
```python
P = (N + 1).float()           # add 1 to every count — "fake" one observation of each bigram
P /= P.sum(dim=1, keepdim=True)
```
This is called **Laplace smoothing** or **add-one smoothing**. It slightly shifts all probabilities away from 0 and 1 — a small price to pay for numerical stability.
**Key insight:** `log(0) = -infinity` crashes training. Adding 1 to counts guarantees every bigram has at least probability > 0. This is the standard fix and appears in virtually every probability-based model. The cost is tiny — you're pretending each bigram was seen once even if it wasn't.

---

### Pattern 8 — Probabilities Must Sum to 1 Per Row
**Video:** makemore video, ~30 mins in
**The problem at that moment:** the count table has raw frequencies, not probabilities. To sample "what comes after 'e'?", you need probabilities not raw counts.
**What was observed:** dividing each row by its own sum converts counts to probabilities. Each row now represents a complete probability distribution — all possible outcomes accounted for, summing to 1.
**The pattern:**
```python
P = N.float()
P /= P.sum(dim=1, keepdim=True)  # each row now sums to 1.0
```
**Key insight:** Probabilities sum to 1 because something must always happen next — 100% of outcomes must be accounted for. If they sum to less than 1, the model is "confused" about some portion of cases.

---

### Pattern 9 — keepdim=True When Dividing Rows by Row Sums
**Video:** makemore video, ~32 mins in
**The problem at that moment:** tried `keepdim=False` and the output looked correct at first glance (same first generated name). Only discovered it was wrong by running 10 generations and seeing them diverge.
**What was observed:**
```
keepdim=True  → sum shape: (27, 1) → broadcasts across columns → each ROW divided by its sum ✓
keepdim=False → sum shape: (27,)   → broadcasts across rows    → each COLUMN divided by wrong value ✗
```
The rows still summed to ~1 by coincidence on this dataset, which is why it wasn't obvious immediately.
**The pattern:**
```python
# CORRECT
P /= P.sum(dim=1, keepdim=True)   # (27,1) broadcasts correctly

# WRONG — silent bug, no error, wrong probabilities
P /= P.sum(dim=1, keepdim=False)  # (27,) aligns to columns
```
**Key insight:** Broadcasting aligns from the right. keepdim=False drops the dimension and causes silent wrong results. Always use keepdim=True when normalising rows. Verify by checking that each row sums to 1.0 before trusting any results.

---

### Pattern 10 — Multinomial Sampling vs Greedy Picking
**Video:** makemore video, ~40 mins in
**The problem at that moment:** had probability rows for each character. Now how do you generate the next character? Always picking the highest probability gives the same name every time.
**What was observed:** `torch.multinomial` samples an index randomly but weighted by probabilities — higher probability = more likely to be picked, but not guaranteed. This produces different names each run while still reflecting what the model learned.
**The pattern:**
```python
# greedy — always same result
next_idx = P[current_idx].argmax()

# multinomial — different each time, weighted by probability
next_idx = torch.multinomial(P[current_idx], num_samples=1, replacement=True, generator=g).item()
```
**Key insight:** Greedy is deterministic and boring. Multinomial samples from the learned distribution — same statistical patterns as training data, different output each time. Generation tasks almost always use sampling not greedy.

---

### Pattern 11 — Generator Scope Controls Reproducibility
**Video:** makemore video, ~42 mins in
**The problem at that moment:** moved generator inside the loop to "randomise" each name. Got the same name 10 times. The generator was being reset to the same initial state every iteration.
**What was observed:**
```python
# WRONG — same name every time
for _ in range(10):
    g = torch.Generator()   # resets to same state each loop
    ...

# CORRECT — different name each time
g = torch.Generator()       # created once, state advances
for _ in range(10):
    ...
```
**The pattern:** Create the generator once outside the loop. Its internal state advances with each sample, producing different results each iteration.
**Key insight:** A Generator created with no seed gives different results each run. With `manual_seed(n)` it gives identical results every run — useful for debugging, bad for actual generation.

---

### Pattern 12 — Negative Log Likelihood as Loss for Language Models
**Video:** makemore video, ~55 mins in
**The problem at that moment:** the counting model assigns probabilities — but how do you measure how good those probabilities are? Need a single number that says "how wrong is this model?"
**What was observed:**
```
model assigned probability P to the correct next character
loss = -log(P)

P = 1.0  (perfect prediction)  → loss = 0
P = 0.5  (ok prediction)       → loss = 0.69
P = 0.01 (very wrong)          → loss = 4.6
```
**The pattern:**
```python
loss = -torch.log(P[xs, ys]).mean()
#                 ↑
#       look up the probability the model assigned
#       to each actual next character in the dataset
```
**Key insight:** NLL directly penalises the model for assigning low probability to the correct answer. Minimising it = model learns to assign high probability to what actually appears in the data. Random baseline for 27 characters = log(27) ≈ 3.3. If loss is near 3.3 after training, nothing was learned.

---

### Pattern 12.5 — .item() to Convert a Single-Element Tensor to a Python Number
**Video:** makemore video, sampling section
**The problem at that moment:** `torch.multinomial` returns a tensor even when sampling just 1 index. You cannot use a tensor as a dictionary key to look up `itos[idx]`.
**What was observed:**
```python
idx = torch.multinomial(P[ix], num_samples=1)
# idx is tensor([13]) — a tensor, not an integer

itos[idx]   # TypeError — can't use tensor as dict key

idx = torch.multinomial(P[ix], num_samples=1).item()
# idx is 13 — a plain Python int

itos[idx]   # works
```
**Key insight:** `.item()` extracts the single value from a 1-element tensor into a plain Python scalar. Use it whenever you need to use a tensor value as a Python int or float — dictionary lookup, `range()`, comparisons, printing a single number. Only works on tensors with exactly 1 element.

---

### Pattern 13 — One-Hot Encoding: Characters as Vectors
**Video:** makemore video, neural net section, ~1:05
**The problem at that moment:** switching from the counting model to a neural net — the net needs numerical input, but you can't feed raw integers because 'm'(13) being "bigger" than 'e'(5) is meaningless.
**What was observed:** represent each character as a vector of 27 zeros with a single 1 at that character's index. Now all characters are equally "different" from each other — no false numerical relationships.
**The pattern:**
```
'e' (index 5) → [0, 0, 0, 0, 0, 1, 0, 0, ..., 0]
                  0  1  2  3  4  5  6  7      26
```
**Key insight:** One-hot removes the false ordering that integer encoding implies. The tradeoff is sparsity — 27 dimensions, only 1 non-zero. Embeddings (Pattern 14) solve the sparsity problem but come later.

---

### Pattern 14 — Matrix Multiply as a Differentiable Row Lookup
**Video:** makemore video, ~1:08
**The problem at that moment:** had a one-hot vector for the current character and a weight matrix W (27×27). The operation `xenc @ W` looked like complex math but felt like it should just be a lookup.
**What was observed:** multiplying a one-hot vector by W zeroes out every row except the one corresponding to the active character. The result is exactly that row of W — nothing more.
```
'e' one-hot: [0, 0, 0, 0, 0, 1, 0, ..., 0]
             × W (27×27 weights)
           = row 5 of W   ← weights for input 'e'
```
**The pattern:** `one_hot @ W` = index into W = `W[character_index]`. The matrix multiply is just a differentiable, batchable way to do a lookup.
**Key insight:** This is why embedding layers exist — they skip the one-hot entirely and just index the row directly (`W[idx]`). Same result, more efficient. The matrix multiply and the embedding lookup are mathematically identical.

---

### Pattern 14.1 — The Trained Bigram Model IS the W Matrix
**Source:** makemore bigram neural network
**The problem at that moment:** after training, what exactly is the model? What does inference use?
**What was observed:** the entire trained bigram model is the 27×27 weight matrix W. Nothing else. Every row represents one input character and contains the learned weights for all 27 possible next characters:
```
Row 0  = weights for input '.'  → what comes after start token
Row 5  = weights for input 'e'  → what comes after 'e'
Row 13 = weights for input 'm'  → what comes after 'm'
```
Inference is just:
```
current char → look up its row in W → softmax → probabilities → sample next char → repeat
```
**The connection to the counting model:**
The W matrix and the counting table learn the same thing — the probability distribution of next characters given the current one. The counting model does it by dividing frequencies. The neural net does it through backprop. Both converge to the same answer given enough data.

**The limitation — no context memory:**
Each prediction depends only on the current character. 'e' always produces the same probability distribution regardless of what came before — "emm" or "ale" makes no difference. The model has no memory of prior characters.
This is why MLP (makemore Part 2) extends the context window, and why transformers eventually look at the full sequence. Every step up is solving this one limitation.

**Key insight:** For the bigram model, the trained model = one matrix. For deep networks, the model = many matrices (one per layer). The inference pattern is the same — pass input through learned weights, sample from output probabilities.

---

## Project 2 — Rebuild MLP in PyTorch

---

### Pattern 14.5 — The PyTorch Neural Network Mental Model (High Level)
**Project:** Rebuild MLP in PyTorch
**The problem at that moment:** what is the complete mental model for building and training any neural network in PyTorch?
**The pattern:**

**1. Define a reusable MLP foundation**
Build a generic `MLP` class using `nn.Module` and `nn.Linear`. This is your reusable building block — it knows nothing about your specific problem.

**2. Instantiate it for your problem**
```python
model = MLP([13, 16, 16, 3])  # 13 inputs, two hidden layers, 3 outputs
```
Sometimes you subclass MLP for custom behaviour, but often just instantiating with the right sizes is enough.

**3. Prepare input and output tensors**
```python
X  # input tensor  — shape (num_examples, num_features), dtype float32
y  # output tensor — shape (num_examples,), dtype long for classification
```

**4. Define a learning rate**
```python
learning_rate = 0.01
```

**5. Training loop — always these steps in this order:**
```python
for step in range(n_steps):
    pred = model(X)                              # 5.1 forward pass
    loss = F.cross_entropy(pred, y)              # 5.1 loss calculation (raw logits, not softmax)

    for param in model.parameters():
        param.grad = None                        # 5.2 zero gradients

    loss.backward()                              # 5.3 backward pass

    for param in model.parameters():
        param.data -= learning_rate * param.grad # 5.4 update: opposite direction of grad
```

**Loss function choice:**
- Classification → `F.cross_entropy(pred, y)` — includes softmax internally, pass raw logits
- Regression → `F.mse_loss(pred, y)` or `((pred - y) ** 2).mean()`

**Key insight:** Steps 5.1–5.4 never change regardless of the problem. Only the model architecture, loss function, and data change. Internalise this loop — it is the foundation of all neural network training.

---

### Pattern 15 — Neuron is Implicit in PyTorch
**Project:** Rebuild MLP in PyTorch
**The problem at that moment:** micrograd required building Neuron, then Layer, then MLP. In PyTorch there is no `Neuron` class to write.
**What was observed:** `nn.Linear(n_inputs, n_outputs)` replaces both `Neuron` and `Layer` at once. It internally holds a weight matrix `(n_outputs, n_inputs)` and a bias vector `(n_outputs,)` — equivalent to `n_outputs` neurons each with `n_inputs` weights.
```python
# micrograd — explicit
class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(...) for _ in range(n_inputs)]
        self.b = Value(...)

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

# PyTorch — neuron is implicit
layer = nn.Linear(n_inputs, n_outputs)  # same thing, built-in
```
**Key insight:** In PyTorch you start at the Layer level. `nn.Linear` is your atomic building block, not `Neuron`. The individual neurons still exist mathematically — they're just not exposed as objects. And because the neuron is implicit, so is everything inside it: the multiplication of each input by its weight, the summation, and the bias addition all happen inside `nn.Linear` without you writing any of it.

---

### Pattern 16 — `forward` is Required; `model(x)` Calls It
**Project:** Rebuild MLP in PyTorch
**The problem at that moment:** micrograd used `__call__` to define how data flows through the network. In PyTorch the method is named differently.
**What was observed:** `nn.Module` wires `__call__` to `forward` internally. You define `forward`, PyTorch calls it when you do `model(x)`:
```python
# micrograd
class MLP:
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# PyTorch — same thing, different method name
class MLP(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model(x)  # → triggers nn.Module.__call__ → calls forward(x)
```
You never call `model.forward(x)` directly — always `model(x)`. PyTorch adds hooks and other machinery in `__call__` that you'd miss by calling `forward` directly.
**Key insight:** `forward` is the PyTorch equivalent of `__call__` in micrograd. Define it, never call it directly. `model(x)` is always the right way.

---

### Pattern 17 — No Activation on the Last Layer
**Project:** Rebuild MLP in PyTorch
**The problem at that moment:** micrograd applied tanh on every neuron including the last layer. Is this correct in general?
**What was observed:** tanh squashes output to (-1, 1). That worked for the toy data because targets were `[1.0, -1.0, -1.0, 1.0]`. But in general:
- Classification: last layer needs raw logits for `F.cross_entropy` — tanh would crush them
- Regression: output could be any value — tanh limits range to (-1, 1)
```python
def forward(self, x):
    for layer in self.layers[:-1]:
        x = torch.tanh(layer(x))  # activation between layers
    return self.layers[-1](x)     # no activation on last layer — raw output
```
**Key insight:** Activation functions between layers create non-linearity. The last layer produces raw output that the loss function interprets. Applying tanh on the last layer limits what values the model can predict. Convention: activation on all hidden layers, none on the output layer.

---

### Pattern 18 — Inherit from `nn.Module`, Store Layers in `nn.ModuleList`
**Project:** Rebuild MLP in PyTorch
**The problem at that moment:** tried inheriting directly from `nn.ModuleList`. Slicing `self[:-1]` in `forward` triggered `MLP.__init__` with wrong arguments and crashed.
**What was observed:** `nn.ModuleList` slicing creates a new instance of whatever class it is. Since `MLP` subclassed `ModuleList`, slicing tried to call `MLP.__init__` with the sliced modules — breaking it. The standard pattern is to inherit from `nn.Module` and hold a `ModuleList` as an attribute:
```python
# WRONG — inheriting from ModuleList causes slice bug
class MLP(nn.ModuleList):
    def forward(self, x):
        for layer in self[:-1]:  # crashes — tries to call MLP.__init__
            ...

# CORRECT — inherit from Module, store list as attribute
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([...])

    def forward(self, x):
        for layer in self.layers[:-1]:  # works fine
            ...
```
**Key insight:** Always inherit from `nn.Module`. `nn.ModuleList` is a container you store inside a Module, not a base class to inherit from. This is the standard pattern in all PyTorch code.

---

## Chapter 1 — fast.ai "Practical Deep Learning for Coders"

---

### Pattern 19 — Overfitting is the Central Challenge
**Source:** fast.ai Chapter 1
**The problem at that moment:** the model performs well on training data but fails on new data — it memorised the training examples instead of learning the underlying pattern.
**What was observed:** overfitting is not a rare edge case — it is the default outcome if you train long enough on any dataset. Every model will eventually overfit given enough steps.
```
training loss:   keeps going down ↓
validation loss: goes down, then turns and goes up ↑  ← overfitting starts here
```
The fix is a **validation set** — data held out from training that you evaluate on after each epoch. The moment validation loss stops improving, stop training. Without a validation set you have no way to detect overfitting at all.

**Why a separate validation set is a must:**
- Training loss always goes down — it tells you nothing about generalisation
- Test set is for final evaluation only — if you use it to tune, it becomes another training set
- Validation set is the early warning system between training and final test

**Key insight:** Overfitting is the single most important and challenging problem in ML. Every technique you will learn — dropout, regularisation, data augmentation, early stopping, batch norm — exists primarily to fight overfitting. Always have a validation set. Always watch validation loss, not just training loss.

---

### Pattern 20 — Always Start with a Pretrained Model
**Source:** fast.ai Chapter 1
**The problem at that moment:** should you train a model from scratch or start from a pretrained model?
**What was observed:** pretrained models have already learned general representations from millions or billions of examples. Fine-tuning them for your task requires a fraction of the data, compute, and time compared to training from scratch.
```
Training from scratch:    random weights → learn everything from your data
                          needs: huge dataset, huge compute, weeks of training

Fine-tuning pretrained:   weights already encode useful features → adapt to your task
                          needs: small dataset, modest compute, hours of training
```
**Why this matters for your roadmap:**
- Your LoRA fine-tuning of Llama 3.2 3B is exactly this — borrowing billions of examples of language understanding, then adapting to home automation with a few thousand examples
- CLIP for robot navigation (Month 2) — borrowing 400 million image-text pairs for free
- MobileNet for webcam classification (Month 1) — borrowing ImageNet training for free

**When to NOT use a pretrained model:**
- Your data is so different from anything the pretrained model saw that its weights are actively harmful (rare)
- You are doing foundational research on new architectures (OpenAI, DeepMind — not you right now)

**Key insight:** Nobody trains from scratch anymore except the top 5-10 frontier labs. Pretrained models are the foundation of all practical ML. The question is never "should I use a pretrained model?" — it's "which pretrained model and how do I fine-tune it?"

---

## Week 3-4 — fast.ai + CNN Projects

---

### Pattern 21 — DataLoader Has No Batches When bs > Dataset Size
**Project:** Spill classifier overfit experiment
**The problem at that moment:** `dls.one_batch()` threw `ValueError: This DataLoader does not contain any batches` even though `len(dls.train_ds)` showed 8 samples.
**What was observed:** fast.ai's default batch size is 64. With 8 training samples and `drop_last=True` (default), `8 // 64 = 0` complete batches. No error during DataLoader creation — only fails at first batch access.
**Fix:**
```python
dls = spill_persona.dataloaders(path, bs=4)  # always pass bs explicitly with small datasets
```
**Key insight:** Always pass `bs` explicitly when working with small datasets. The default of 64 silently produces 0 batches if your dataset is smaller. No warning, just a runtime error on first use.

---

### Pattern 22 — aug_transforms() Causes MPS Error on Mac (Perspective Warp)
**Project:** Spill classifier overfit experiment — augmentation phase
**The problem at that moment:** `aug_transforms()` caused `NotImplementedError: aten::_linalg_solve_ex not implemented for MPS`. The error appeared during DataLoader creation, not training.
**What was observed:** `aug_transforms()` includes a perspective warp transform that calls a linear algebra operation not supported on Apple MPS GPU. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` mid-session doesn't work — must be set before PyTorch is imported.
**Fix:**
```python
batch_tfms=aug_transforms(max_warp=0)  # disables perspective warp, all other augmentations still apply
```
**Key insight:** `max_warp=0` disables the one transform causing the MPS failure. If you need the full env var fix, set it before launching Jupyter: `PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter notebook`. Never set `fastai.torch_core.default_device` to a device object — it breaks the session and requires kernel restart.

---

### Pattern 23 — Random Split With Too Few Images Per Class Drops a Class Entirely
**Project:** Spill classifier overfit experiment
**The problem at that moment:** `KeyError: "Label 'Reflection' was not included in the training dataset"` when calling `show_batch()`. Dataset had 2 images per class, valid_pct=0.5.
**What was observed:** With 2 images per class and 50% split, random chance put both Reflection images in validation and none in training. The CategoryBlock builds its vocabulary from the training set only — unseen labels in validation cause a KeyError.
**Fix:** Use at least 3-4 images per class so random splits reliably put at least 1 in each split. Changing the seed is a fragile workaround.
**Key insight:** Minimum viable images per class for a 50/50 split is ~4. Below that, random splits routinely strand entire classes in one split. The error only appears at `show_batch` or inference time, not during DataLoader creation.

---

### Pattern 24 — Mini-Batches: Why Not Full Dataset or Single Item
**Source:** fast.ai Chapter 4 — MNIST digit classifier
**The problem at that moment:** understanding why SGD uses mini-batches instead of the full dataset or one image at a time
**What was observed:** full dataset = 1 weight update per epoch; single item = noisy unstable gradient; mini-batch = best of both
**The pattern:**
- Full dataset per update: accurate gradient but only 1 update per epoch — model learns slowly
- Single item per update: many updates but gradient is noisy and unstable — learning is erratic
- Mini-batch (64 images): ~187 updates per epoch on MNIST, gradient stable enough to be meaningful
- More frequent updates = model already significantly corrected before epoch 1 even ends
- Noise from different random batches acts as a regulariser — forces weights to work across many subsets, not just memorise the full training set
- Mini-batches slow down overfitting but do not prevent it — you can still overfit with mini-batches over many epochs (seen in spill classifier at epoch 26)
**Key insight:** mini-batches give you more weight updates per epoch for the same compute cost, and the randomness accidentally improves generalisation. Start with batch size 64 for image tasks.

---

### Pattern 25 — Batch Size and Learning Rate Move Together
**Source:** fast.ai Chapter 4 — MNIST digit classifier
**The problem at that moment:** understanding what batch size to choose and whether it matters beyond speed
**What was observed:** batch size and learning rate are not independent — changing one without the other changes how the model learns
**The pattern:**
- Smaller batch (16-32): noisier gradient → better generalisation, more Python overhead per epoch
- Larger batch (128-512): smoother gradient → faster wall-clock time, but tends to overfit more
- Empirical rule: if you double the batch size, halve the learning rate to get similar generalisation
- Changing batch size between experiment runs makes results incomparable — keep it fixed per project
- Default starting point: batch size 64, tune learning rate first before touching batch size
**Key insight:** batch size and learning rate are a pair — the generalisation behaviour of a model depends on their ratio, not on either value alone.

---

## Index — Find Pattern by Problem

| I need to... | Pattern |
|---|---|
| Understand why a number needs to be wrapped in Value | Pattern 0a — Value Object |
| Understand what the computation graph is | Pattern 0b — Computation Graph |
| Understand how gradients flow through a chain of operations | Pattern 0b.5 — Chain Rule |
| Understand how each operation knows its gradient rule | Pattern 0c — Each Op Defines Its Own Backward |
| Know the actual math behind each gradient rule | Pattern 0c.1 — Local Derivative Formulas |
| Understand why _backward is defined inside each operation | Pattern 0c.2 — Closures |
| Understand why gradients use += not = | Pattern 0d — Gradient Accumulation |
| Understand why loss.grad starts at 1.0 | Pattern 0d.5 — Backprop Seed |
| Understand the order backprop visits nodes | Pattern 0e — Topological Sort |
| Fix TypeError when using Value with Python sum() | Pattern 0f — __radd__ |
| Understand why weights must be random not equal | Pattern 0g — Random Initialisation |
| Fix TypeError when using ** on a Value | Pattern 0h — __pow__ |
| Understand why stacking layers works | Pattern 1 — Activation Functions |
| Understand what bias is for | Pattern 2 — Bias |
| Know which direction to move weights | Pattern 3 — Gradient Direction |
| Fix corrupted gradients between steps | Pattern 4 — Zero Gradients |
| Structure the training code | Pattern 5 — Training Loop |
| Iterate over consecutive character pairs | Pattern 5.5 — zip sliding window |
| Store bigram frequencies for fast lookup | Pattern 6 — 2D Lookup Table |
| Convert characters to numbers and back | Pattern 7 — stoi and itos |
| Avoid wrong results when dividing integer tensors | Pattern 7.5 — .float() before division |
| Avoid log(0) = infinity crashing training | Pattern 7.6 — Smoothing |
| Convert counts to probabilities | Pattern 8 — Row Normalisation |
| Normalise rows not columns | Pattern 9 — keepdim=True |
| Generate varied output from a model | Pattern 10 — Multinomial Sampling |
| Control reproducibility of random sampling | Pattern 11 — Generator Scope |
| Measure how wrong a language model is | Pattern 12 — Negative Log Likelihood |
| Use a tensor value as a Python int or float | Pattern 12.5 — .item() |
| Feed characters into a neural network | Pattern 13 — One-Hot Encoding |
| Understand what xenc @ W is doing | Pattern 14 — Matrix Multiply as Lookup |
| Understand what the trained bigram model is + why it's limited | Pattern 14.1 — W Matrix IS the Model |
| Get the complete mental model for building any PyTorch NN | Pattern 14.5 — PyTorch Mental Model |
| Understand why there is no Neuron class in PyTorch | Pattern 15 — Neuron is Implicit |
| Understand what forward() is and how model(x) works | Pattern 16 — forward and model(x) |
| Know whether to apply activation on the last layer | Pattern 17 — No Activation on Last Layer |
| Fix crash when slicing layers in forward() | Pattern 18 — nn.Module vs nn.ModuleList |
| Understand why overfitting is the central challenge in ML | Pattern 19 — Overfitting and Validation Set |
| Know when to use a pretrained model vs train from scratch | Pattern 20 — Pretrained Models |
| Fix zero batches error with small datasets in fast.ai | Pattern 21 — DataLoader bs |
| Fix MPS error with aug_transforms on Mac | Pattern 22 — aug_transforms MPS |
| Fix missing class error with small datasets and random split | Pattern 23 — Random Split Class Drop |
| Understand why mini-batches are better than full dataset or single item | Pattern 24 — Mini-Batches |
| Know what batch size to use and how it interacts with learning rate | Pattern 25 — Batch Size and Learning Rate |
