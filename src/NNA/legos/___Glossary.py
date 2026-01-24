""" #TODO automate this
        GlossaryIndex = {
        "gradient": Glossary_Blame_Signal,
        "epoch": Glossary_Epoch,
        "sgd": Glossary_SGD,
        "loss": Glossary_LossFunction,
        "mse": Glossary_MSE,
        "activation": Glossary_Activation,
        "relu": Glossary_ReLU,
        "softmax": Glossary_Softmax,
        "weight_init": Glossary_WeightInit,
        "learning_rate": Glossary_LearningRate,
        "backpropagation": Glossary_Backprop,
        "activation": Glossary_ActivationOut,
        "input_activation": Glossary_ActivationIn,
    }
"""


class Glossary:
    """
    👁️ We think in language… and if that language is not clear, neither is our thinking.
    🧠 This defines alternative terms that are more accurate, explainable, and humane.
    🥂 Each entry replaces a misleading legacy term with a transparent, honest one.

    temp list for window...
    Spreading Blame from Neuron 20 to Neuron 10
    Influence.  How much of Neuron 20’s output was caused by Neuron 10?"
    My Share:   Based on influence how much actual blame is received.
    MBFA:       My blame from all.
    Accepted:   MBFA * Act Grad

    """
    def __init__(self, old_name, new_name, why_old_name_sucks="Because", definition="Please add"):
        self.old_name = old_name
        self.new_name = new_name
        self.why_old_name_sucks = why_old_name_sucks
        self.definition = definition

    def __repr__(self):
        return f"Glossary Entry: '{self.old_name}' → '{self.new_name}'"

Glossary_RawAdjustment = Glossary(
    old_name="Gradient",
    new_name="Raw Adjustment",
    why_old_name_sucks="""
        'Gradient' is a borrowed metaphor from calculus that implies slopes, geometry, and smooth manifolds —
        concepts that don’t apply cleanly in the high-dimensional, discrete world of neural networks.
        Worse, it masks what the value *actually is*: a direct precursor to a weight change.
    """,
    definition="""
        The Raw Adjustment is the value that tells a weight how it should change to reduce error —
        before it's scaled by the learning rate.

        It is calculated per weight during backpropagation:
            Raw Adjustment = Input × Accepted Blame

        The learning rate turns this into the Final Adjustment:
            Final Adjustment = Learning Rate × Raw Adjustment
    """
)

Glossary_IncomingBlame = Glossary(
    old_name="Pre-activation error signal",
    new_name="Incoming Blame",
    why_old_name_sucks="""
        Most resources just gloss over this — they leap straight to ‘delta’ or ‘error’.
        But this is a *real number with real meaning*: raw responsibility.
        """,
    definition="""
        Incoming Blame is the total amount of responsibility this neuron receives from downstream neurons.

        It’s a sum of **Blame Shares** sent by all connected children:
        → `Incoming Blame = Σ (Child’s Blame × Weight to this neuron)`

        It represents what the next layer is saying:
        “Here’s how much your actions contributed to our mistake.”
        """
)

Glossary_AcceptedBlame = Glossary(
    old_name="Delta",
    new_name="Accepted Blame",
    why_old_name_sucks="""
        ‘Delta’ is abstract and mathematical. What’s really happening here is a neuron choosing to *own* part of the mistake — based on how ready it was to change.
        """,
    definition="""
        Accepted Blame is how much of the Incoming Blame a neuron agrees to take on.

        It’s the Incoming Blame scaled by how sensitive the neuron was (i.e., its activation gradient):
        → `Accepted Blame = Incoming Blame × Activation Gradient`

        This is what the neuron uses to adjust its own weights — it’s saying:
        “Okay, given how active I was, this is how much of the mistake I’ll take responsibility for.”
        """
)

Glossary_BlameShare = Glossary(
    old_name="Weighted Error" or "Part of the chain rule",
    new_name="Blame Share",
    why_old_name_sucks="""
        Most sources don’t name this step at all, or call it something vague like 'weighted error'.
        But it’s the clearest part of the process: passing on your blame proportionally.
        """,
    definition="""
        After a neuron receives its own blame, it divides that blame among the neurons that fed into it.

        This process is based on how much influence each input had.  
        → `Blame Share = My Blame × Input Weight to that neuron`

        Think of blame as a pie:  
        This neuron slices it up based on who helped it make the mistake.
        Each upstream neuron receives their **Blame Share**.
        """
)


Glossary_ChainRule = Glossary(
    old_name="Chain Rule",
    new_name="Blame Cascade",
    why_old_name_sucks="""
        'Chain rule' makes it sound like dry calculus. In backprop, it’s how blame travels.
        Without it, you'd never know how upstream decisions affected downstream results.
        """,
    definition="""
        The Blame Cascade shows how small changes early in the network impact the final result.
        It's like tracing back: “If you hadn’t done that, we wouldn’t have ended up here.”

        It works by multiplying how much each layer influenced the next,
        allowing blame to be fairly split across multiple contributors.
        """
)
Glossary_GD = Glossary(
    old_name="Gradient Descent",
    new_name="GBS",
    why_old_name_sucks="""
        'Descent' implies a hill, but this isn’t geography. There’s no landscape, just numbers.
        It hides how indirect and unstable the process really is.
        """,
    definition="""
        This strategy changes weights by following the direction of increasing blame.
        It doesn’t know where the best settings are — it just reacts to how bad things are right now.

        Think of it as adjusting dials to make the pain stop.
        It can work — but it’s not graceful.
        """
)


Glossary_Accusation = Glossary(
    old_name="There is no term for this",
    new_name="Accusation",
    why_old_name_sucks="""
        Each child neuron (on the next layer) sends a portion of its blame back to the current neuron.

        That portion is: Child’s Blame×Weight that connects us
        That weight is the one from current neuron → child neuron
        So in child’s frame, it’s: "Here’s how much you influenced me, scaled by how much I’m hurting."

        🧠 Traditional Term?  In backpropagation literature, this is usually unnamed — it’s just a step inside the chain rule.
            But if you do find a name, it might show up as:
            “Weighted error” (but vague and overloaded)
            “Backpropagated gradient” (which is just... nonsense for clarity)
            Or most often: nothing. It's treated as an anonymous middle step.
        """,
    definition="""
        An Accusation is an incoming blame value passed from a neuron in the next layer (a child) to this neuron.
        ???Blameshare from the opposite perspective?
        Each child neuron says:  
        “Part of my error is your fault — here’s how much you influenced me.”

        Accusation is calculated as:
        → `BlameFromChild × InfluenceFromThisNeuron`

        Multiple accusations can be received from multiple children.
        These are summed into **My Blame From All (MBFA)** — the total raw blame this neuron is being assigned.
        """
)

Glossary_BatchTotal = Glossary(
    old_name="N/A (no consistent term for this in traditional ML)",
    new_name="Batch Total",
    why_old_name_sucks="""
        There isn’t even a standard name for this — which is wild, considering how critical it is.
        Most frameworks quietly accumulate gradients across a batch without ever surfacing them.
        That hides the connection between per-sample adjustments and the final weight change.
        """,
    definition="""
        Batch Total is the **sum of raw adjustments across all samples in the current batch** for a given weight.

        It answers the question:
        “Across this entire batch, how much are we being told to adjust this weight?”

        Raw Adjustment = Input × Accepted Blame  
        Batch Total    = Σ Raw Adjustments (one per sample)

        This value is what ultimately gets scaled by the learning rate to produce the final weight update.

        → Final Adjustment = Batch Total × Learning Rate

        NeuroForge shows this explicitly to maintain full auditability in batch training modes — no math hidden behind the curtain.
        """
)



Glossary_Activation = Glossary(
    old_name="Activation",
    new_name="Output",
    why_old_name_sucks="""
        'Activation' is a biology metaphor gone rogue.
        It implies something binary or mysterious. But in neural networks, it’s just the output value of a neuron — the signal it sends forward.
        """,
    definition="""
        The 'activation' is simply the result a neuron sends to the next layer.
        After summing its inputs and applying a transformation (like ReLU), the neuron outputs a number.
        That number becomes input to the next layer.
        So let’s call it what it is: the neuron’s output.
        """
)


Glossary_ActivationIn = Glossary(
    old_name="Input Activation",
    new_name="Input",
    why_old_name_sucks="""
        Stacking 'input' and 'activation' adds noise to a clean idea.
        It's just the number this neuron receives — from its parents.
        Calling it an 'activation' before it’s even been activated makes no sense.
        """,
    definition="""
        This is the value a neuron receives from another neuron in the previous layer.
        You might have 3 parents, each sending you a number — those are your inputs.
        After combining them, you decide what to output.
        """
)


Glossary_Epoch = Glossary(
    old_name="Epoch",
    new_name="Full Data Pass",
    why_old_name_sucks="The term 'epoch' is inherited from timekeeping and means nothing to most learners. It hides what actually happens: one complete loop over all training data.",
    definition="""
        An epoch is a full pass through the entire training dataset.
        Every sample is shown to the model exactly once.
        We often repeat this multiple times to improve learning.
        """
)

Glossary_SGD = Glossary(
    old_name="Stochastic Gradient Descent",
    new_name="Random Update Trick",
    why_old_name_sucks="""
        It's not stochastic if you're using a full batch.
        It’s not gradient descent in any continuous sense.
        And it hides the fact that the only reason it works is because of brute force.
        A nudge in the right direction is all that is needed.
        """,
    definition="""
        This strategy updates weights based on one or a few samples at a time.
        It adds noise and chaos to the learning process, which sometimes prevents bad patterns from forming.
        But it’s less about theory, and more about getting lucky with random steps.
        """
)


Glossary_LossFunction = Glossary(
    old_name="Loss Function",
    new_name="Gradient Source",
    why_old_name_sucks="""
        It's focus in formal teaching implies the value matters — it doesn’t.
        Only the derivative of the loss (i.e., the gradient) affects backpropogation.
        Models don’t ‘look’ at loss, they respond to the signal behind it.
        """,
    definition="""
        A gradient source tells the network how to respond to its mistake.
        It doesn't matter how big the loss number is — only which direction to nudge each weight.
        Loss functions are like signposts. You ignore the post. You follow the arrow.
        """
)


Glossary_MSE = Glossary(
    old_name="Mean Squared Error",
    new_name="Squared Error Signal",
    why_old_name_sucks="""
        MSE is not really about squaring anything for training — the model only sees 2 * error.
        The square is make this better
        """,
    definition="""
        The gradient of MSE is just 2 times the error.
        The ‘squared’ part makes bigger mistakes count more — that’s all.
        It’s not more precise, just more punishing.
        """
)


Glossary_Activation = Glossary(
    old_name="Activation Function",
    new_name="Output Transformer",
    why_old_name_sucks="""
        'Activation' makes it sound biological and binary — like a neuron turns on or off.
        But in most networks, these functions just reshape the neuron’s output.
        """,
    definition="""
        This function transforms the output of a neuron.
        It can compress it (like sigmoid), sharpen it (like ReLU), or center it (like tanh).
        It’s not turning anything ‘on’ — it’s just shaping how signals flow.
        """
)


Glossary_ReLU = Glossary(
    old_name="ReLU (Rectified Linear Unit)",
    new_name="Signal Gate",
    why_old_name_sucks="""
        The acronym 'ReLU' is meaningless to beginners, and 'unit' makes it sound like a separate entity.
        There's no 'rectifier' or 'unit' — just a rule that blocks negative numbers.
        """,
    definition="""
        ReLU lets positive values through and blocks negative ones.
        It's like a gate: if the signal is positive, pass it; if not, shut it down.
        Most networks use ReLU to force neurons to contribute *only when excited.*
        """
)


Glossary_Softmax = Glossary(
    old_name="Softmax",
    new_name="Probability Estimator",
    why_old_name_sucks="""
        'Softmax' sounds technical but tells you nothing.
        It doesn’t soften anything or maximize anything in human terms.
        It just squishes numbers into a probability distribution.
        """,
    definition="""
        Takes a list of values and turns them into a set of probabilities that add up to 1.
        Higher numbers become more likely — but everything still gets some chance.
        Used to make the model ‘choose’ one of many possible outcomes.
        """
)


Glossary_WeightInit = Glossary(
    old_name="Weight Initialization",
    new_name="Start Guess",
    why_old_name_sucks="""
        'Initialization' sounds technical and forgettable.
        But your model’s *entire future* is shaped by how it starts.
        This is its first guess — and a bad first guess can sabotage everything.
        """,
    definition="""
        Before training starts, every weight needs a starting value.
        That starting value can be random, balanced, zeroed, or carefully scaled.
        It’s the model’s first take — and it matters more than most people realize.
        """
)


Glossary_LearningRate = Glossary(
    old_name="Learning Rate",
    new_name="Nudge Size",
    why_old_name_sucks="""
        It’s not a rate. It’s not about learning. It’s a multiplier on blame.
        ‘Learning rate’ makes it sound like intelligence. It’s just how big the updates are.
        """,
    definition="""
        This is how strongly the model reacts to blame.
        A big nudge makes it learn fast but risks overshooting.
        A small nudge is stable but can be slow or stuck.
        Tuning this is like finding the right volume for a microphone: too low = nothing, too high = feedback.
        """
)


Glossary_Backprop = Glossary(
    old_name="Backpropagation",
    new_name="Blame Relay",
    why_old_name_sucks="""
        'Backpropagation' sounds like a quantum physics term.
        It hides the simple truth: blame is passed backward through the network to assign responsibility.
        """,
    definition="""
        After making a prediction, the network looks at how wrong it was.
        It then passes that blame backward through each layer,
        so every neuron knows how much it contributed to the error.
        This is how learning happens: every weight gets its share of the blame.
        """
)


"""
🎬 The Family Movie Metaphor
Let’s break it down in your words — and fill in the labels as we go:

🎥 Forward Prop = Decision Phase
Children (inputs) toss out ideas:
"Let’s watch a Marvel movie!"
"How about something animated?"

Parents (middle layer) weigh the suggestions:
"Hmm... maybe Spider-Verse? Or Elemental?"
They combine the kids’ preferences with their own biases (weights), then...

Grandparents (output layer) make the final call:
"Alright, we’ll go with Spider-Verse."

The signal has moved forward through the family tree — each generation processing what they received.

🔁 Backward Prop = Blame Phase
They watch the movie... and it’s a disaster. 😩
("Too loud!" "Didn’t understand it!"_)

Grandparents say:
"Why did we agree to this? The parents convinced us!"

Parents go:
"We were just trying to make the kids happy!"

And the blame flows backward — each layer passing responsibility upstream.

🧠 What This Solves
✅ Preserves parent–child direction in forward flow (still accurate structurally)

✅ Justifies blame flowing in reverse during backprop (natural and intuitive)

✅ Makes adjustments feel like “family therapy” after a bad choice 😄

✅ Reinforces the core idea:
"Outputs are built by children → parents → grandparents, but errors are explained by grandparents → parents → children."

💡 Tooltip/Wiki Version:
Forward Pass
“The family decides which movie to watch — kids suggest, parents refine, grandparents decide.”

Backward Pass
“The movie sucked. Grandparents blame parents. Parents blame kids. Everyone adjusts their preferences next time.”

"""



################### UNSURE BELOW THIS POINT
Glossary_ConnectionWeight = Glossary(
    old_name="Weight between neurons",
    new_name="Connection Weight",
    why_old_name_sucks="""
        'Weight' is vague and generic. It doesn't clarify *whose* weight it is, *what direction* it connects, or *why it matters* in both forward and backward flow.
        Most explanations treat weights as static scalars rather than the **dual-purpose conduits** they truly are.
        """,
    definition="""
        A Connection Weight links two neurons: one in the current layer (the sender) and one in the next layer (the receiver).

        It has two critical roles:

        🔹 **Forward Pass (Influence)**:  
        The weight scales the sender neuron's output — determining how much it contributes to the receiver's input.  
        → `Output × Weight → Raw Sum of receiver`

        🔹 **Backward Pass (Accusation Pathway)**:  
        The same weight also determines how much of the receiver’s blame is sent backward to the sender.  
        → `Accepted Blame of receiver × Weight → Accusation sent back to sender`

        Connection Weights are the exact same values used in both directions of learning.  
        - In forward pass, they **control influence**.  
        - In backprop, they **carry blame**.

        NeuroForge tracks and adjusts these weights during training via:
        → `Adjustment = Raw Adjustment × Learning Rate`
        """
)



Glossary_Parent = Glossary(
    old_name="Previous layer neurons",
    new_name="Parents",
    why_old_name_sucks="It’s not wrong, but it’s sterile. Parents *feed* the current neuron, just like in a family tree.",
    definition="Neurons from the layer to the left whose outputs serve as inputs to the current neuron. They ‘parent’ the activation."
)

Glossary_Parent2 = Glossary(
    old_name="Next layer neurons",
    new_name="Parents (Blame-Centric)",
    why_old_name_sucks="The structural view calls these ‘children,’ but in backprop, they assign blame — like a parent scolding their kid.",
    definition="""
        Neurons in the layer to the right (next layer) that receive this neuron’s output.
        But during backpropagation, they become the ‘parents’ — evaluating the decision and assigning blame backward.
        """
)


Glossary_Child = Glossary(
    old_name="Next layer neurons",
    new_name="Children",
    why_old_name_sucks="Again, technically fine, but vague. These are the neurons who receive the current neuron’s output. Children inherit activations.",
    definition="Neurons in the layer to the right who receive the output of the current neuron. If this neuron fires, they’re the ones who react to it."
)

Glossary_Child2 = Glossary(
    old_name="Previous layer neurons",
    new_name="Children (Blame-Centric)",
    why_old_name_sucks="Although they come earlier in structure, these neurons are like kids — their outputs shaped the current neuron’s mistake.",
    definition="""
        Neurons in the previous layer who sent inputs to this neuron.
        During backpropagation, this neuron pushes blame onto them — just like parents blaming their kids for a bad family decision.
        """
)

Glossary_Sibling = Glossary(
    old_name="Neurons in same layer",
    new_name="Siblings",
    why_old_name_sucks="‘Same layer’ is visually descriptive but not intuitive. Siblings process in parallel — same inputs, different roles.",
    definition="Other neurons in the same layer — they receive the same inputs but have different weights, biases, and behaviors."
)