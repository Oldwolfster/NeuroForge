# ═══ ArenaDocPatch.py ═════════════════════════════════════════════════
#🔶 REPLACES the five-file version entirely. 75 arenas loaded.
#🔶 Writes IN PLACE with no backup. Copy your arenas folder first.

from pathlib import Path


class ArenaDocPatch:
    """Rewrite arena class docstrings in place. Generation bodies untouched."""

    ARENA_DIR = r"C:\SynologyDrive\Development\PyCharm\NeuroForge\src\NNA\coliseum\arenas"

    # ══════════════════════════════════════════════════════════════
    # DRIVER
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def run(cls):
        for name in cls.DOCS: cls.patch(Path(cls.ARENA_DIR) / name)

    @classmethod
    def patch(cls, path):
#🔶 NEW — a missing file skips instead of killing the run
        if not path.exists(): return print(f"MISSING  {path.name}")
        text = cls.read(path)
        text = cls.swap_doc      (text, cls.DOCS[path.name])
        text = cls.rename_fields (text, cls.RENAMES.get(path.name, []))
        cls.write(path, text)
        print(f"patched  {path.name}")

    # ══════════════════════════════════════════════════════════════
    # SURGERY
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def swap_doc(cls, text, body):
        """Replace the class docstring; insert one if the class has none."""
        nl         = "\r\n" if "\r\n" in text else "\n"
        head, tail = cls.split_after_class(text)
        if tail.lstrip().startswith('"""'): tail = tail.split('"""', 2)[2]
        else:                               tail = nl + tail
        return head + ('    """' + body + '    """').replace("\n", nl) + tail

    @classmethod
    def split_after_class(cls, text):
        """Everything through the 'class X(BaseArena):' line, and the rest."""
        end = text.index("\n", text.index("(BaseArena):")) + 1
        return text[:end], text[end:]

    @classmethod
    def rename_fields(cls, text, pairs):
#🔶 WAS: single-line matcher. Missed Engineered31's multi-line labels list.
        """Label lists only. Opens on a return/labels line holding '[', closes at ']'."""
        out, live = [], False
        for line in text.splitlines(keepends=True):
            if line.strip().startswith(("return", "labels")) and "[" in line: live = True
            if live:
                for old, new in pairs: line = line.replace(f'"{old}"', f'"{new}"')
                if "]" in line: live = False
            out.append(line)
        return "".join(out)

    # ══════════════════════════════════════════════════════════════
    # IO — newline="" so CRLF files stay CRLF
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def read(cls, path):
        with open(path, encoding="utf-8", newline="") as f: return f.read()

    @classmethod
    def write(cls, path, text):
        with open(path, "w", encoding="utf-8", newline="") as f: f.write(text)


# ══════════════════════════════════════════════════════════════════════
# THE DOCSTRINGS — body only. swap_doc supplies the triple quotes.
# ══════════════════════════════════════════════════════════════════════

ArenaDocPatch.DOCS = {

# ─── BASIC ────────────────────────────────────────────────────────────

"Basic_Baseball_BattingAvg.py": """
    A hitter's last four games -- at-bats and hits in each -- and what he will bat
    next time out. Eight inputs, all of which genuinely matter.

    Lesson:       A target that is a ratio rather than a raw count.
    Truth:        Total Hits / Total AB across the four games, + noise(sigma 0.02), clamped 0..1
    Solvable by:  A single neuron gets close. The division is what it cannot do exactly.
    Watch for:    The four Hits weights positive and the four AB weights negative.
""",

"Basic_Baseball_Hits_Flat.py": """
    Four games of a hitter's box score plus how many at-bats he gets tomorrow.
    Only the two most recent games feed the answer; the older two are there to be
    ignored, along with every home run and walk column.

    Lesson:       Recency. Twelve of the seventeen inputs are dead weight.
    Truth:        Hits5 = round(0.6 x G4 Hits + 0.2 x G3 Hits + 0.2 if G4 HR else 0
                  + uniform(-0.25, 0.25)), clamped to 0..Game 5 AB
    Solvable by:  A single neuron, once it zeroes the dead columns.
    Watch for:    Game 4 H carrying roughly three times the weight of Game 3 H.
""",

"Basic_CarValue_2Inputs.py": """
    What a used car is worth, given how old it is and how far it has been driven.
    Both facts push the price down and neither one rescues the other.

    Lesson:         Two inputs, two negative weights. The clean step after a one-input line.
    Truth:          Car Value = 30000 - 1000 x Years - 50 x Miles(1000s) + noise(sigma 500)
    Loss Function:  Mean squared error.
    Solvable by:    A single neuron.
    Watch for:      Both weights landing negative and the bias near 30000.
""",

"Basic_CircleInSquare.py": """
    Points scattered on a square, labelled by whether they land inside a circle
    drawn in the middle. Nothing about the shape is given away -- only the two
    coordinates.

    Lesson:       The first curved boundary. No straight line separates these.
    Truth:        Inside = 1 when sqrt(X^2 + Y^2) <= 1, points drawn from a 3x3 square
    Solvable by:  One hidden layer.
    Watch for:    The corners. A network that is too small gets those wrong first.
""",

"Basic_CreditScore_Repaid.py": """
    A credit score from 1 to 100, and whether that loan came back. The score is not
    a hint about the answer -- it is the odds, so even a 90 defaults sometimes.

    Lesson:       Classification where the answer is genuinely uncertain. The noise
                  is the lesson, not a defect in the data.
    Truth:        P(Repaid) = Credit Score / 100
    Solvable by:  A single neuron.
    Watch for:    Accuracy topping out at 75 percent. That is the ceiling the coin
                  flips impose -- anything higher is memorization.
""",

"Basic_CreditScore_RepaymentRatio.py": """
    A credit score and the fraction of the loan that came back, as a number rather
    than a yes or no. The same story as the Repaid arena, asked as a regression.

    Lesson:       One dataset posed two ways. Run both and compare what you learn.
    Truth:        Ratio = Credit Score / 100 + noise(sigma 0.1), clamped to 0..1
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight settling near 0.01 and the bias staying near 0.
""",

"Basic_CreditScore_RepaymentRatio_NeedsBias.py": """
    The repayment-ratio arena with a floor under it -- nobody pays back less than
    about half. The line no longer starts at the origin.

    Lesson:       Why a bias term exists. Identical to the plain RepaymentRatio
                  arena in every respect except the intercept.
    Truth:        Ratio = 0.5 + Credit Score / 200 + noise(sigma 0.1), clamped to 0..1
    Solvable by:  A single neuron, but only one with a bias.
    Watch for:    Bias near 0.5, weight near 0.005. Turn the bias off and watch it fail.
""",

"Basic_CustomerChurn_3Inputs.py": """
    Subscribers described by how long they stayed, what they paid, and how often
    they called support -- and whether they left. Almost all of them left.

    Lesson:       A badly imbalanced target. Accuracy looks excellent while the model
                  is doing nothing but answering "churned" every time.
    Truth:        P(Churn) = 0.2 + 0.1 x Tenure + 0.05 x Monthly Spend
                  + 0.15 x Service Calls, never capped at 1
    Solvable by:  A single neuron, though there is little here to learn.
    Watch for:    The churn rate in the data itself before you judge any model.
    Notes:        Monthly Spend alone drives the probability past 1 for most rows, so
                  roughly nine in ten churn. See the review list.
""",

"Basic_CustomerChurn_6Inputs.py": """
    SaaS subscribers scored from 0 to 1 on how likely they are to cancel next month.
    Two of the six columns are pure invention and feed nothing.

    Lesson:       Finding the four columns that matter among six. A gentler version
                  of the Red Herring trap.
    Truth:        Score = 0.6 x Invoice Lateness + 0.4 x (1 - Login Freq)
                  + 0.05 x Support Tickets + 0.01 x Days Since Login
                  + noise(sigma 0.02), clamped to 0..1
    Solvable by:  A single neuron.
    Watch for:    Unused 1 and Unused 2 driving their weights toward zero.
""",

"Basic_DefaultRisk_DTI.py": """
    Income, debt, and the ratio between them, labelled high or low risk. The ratio
    column is handed over already computed -- the point is that it did not have to be.

    Lesson:       What a well-chosen engineered feature is worth. The third column
                  makes the first two nearly redundant.
    Truth:        High Risk above a Debt/Income of 0.5, Low Risk below 0.3,
                  a coin flip in between
    Solvable by:  A single neuron, using the ratio column alone.
    Watch for:    Weight collapsing onto Debt-to-Income Ratio while the other two fade.
    Notes:        The 0.3 to 0.5 band is deliberately unlearnable. Errors cluster there.
""",

"Basic_DiseaseRisk_4Inputs.py": """
    Four routine health readings and a risk score between 0 and 1. Each reading is
    centered on its normal value first, so the interesting rows are the abnormal ones.

    Lesson:       A sigmoid target over a linear combination, with the four inputs
                  on wildly different scales.
    Truth:        Risk = sigmoid(0.05 x (BP-120) + 0.07 x (Sugar-100)
                  + 0.10 x (BMI-25) + 0.03 x (Age-50) + noise(sigma 0.5))
    Solvable by:  A single neuron with a sigmoid output.
    Watch for:    Blood Sugar dominating once range is accounted for, despite BMI
                  carrying the larger coefficient.
""",

"Basic_EnergyOutput_Turbine.py": """
    A wind turbine's power output from the weather around it and the machine's own
    settings. Wind speed does nearly all the work, and it does it as a cube.

    Lesson:       A dominant cubic term with three smaller multiplicative corrections
                  layered on top of it.
    Truth:        Power = 0.000306 x WindSpeed^3, multiplied by an RPM/temperature
                  penalty, a blade-stall factor above 30 degrees, and a humidity
                  ripple; noise scales with output
    Solvable by:  One hidden layer, or a single neuron if you cube the wind yourself.
    Watch for:    Error growing with output size. The noise is proportional, not fixed.
""",

"Basic_FuelCost_2Inputs.py": """
    A trip's fuel bill from the distance driven and the price at the pump. The car's
    efficiency is fixed at 25 mpg and never appears as a column.

    Lesson:       A target built by multiplying two inputs rather than adding them.
    Truth:        Fuel Cost = (Miles / 25) x Gas Price + uniform(-1, 1)
    Solvable by:  One hidden layer. A single neuron cannot multiply two inputs.
    Watch for:    A linear fit landing on the average and missing both ends.
    Notes:        Fixed mpg, no traffic, no vehicle variation.
""",

"Basic_HouseValue_1Input.py": """
    House price from square footage and nothing else. The noise is large enough that
    individual houses scatter widely around a line that is nevertheless exactly right.

    Lesson:       One input, one weight, heavy noise. Shows what a correct fit looks
                  like when the data itself is loose.
    Truth:        Price = 200 x Square Feet + noise(sigma 50000)
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight settling near 200 while per-row error stays large.
                  Both of those are the right outcome.
    Notes:        Square footage runs 1500 to 4000, despite an old comment saying 500.
""",

"Basic_Income_1Input_Dollars.py": """
    Salary from years on the job, in whole dollars -- the units a payroll system
    would actually use.

    Lesson:       Input and target magnitude. Nothing about the relationship changes,
                  but the distances an optimizer has to travel do.
    Truth:        Salary = 30000 + 4000 x Years Exp + noise(sigma 5000), floored at 0
    Solvable by:  A single neuron with a bias.
    Watch for:    Whether it trains at all without a scaler. Compare against
                  Basic_Income_1Input_Thousands, the same generator divided by 1000.
""",

"Basic_Income_1Input_NeedsBias.py": """
    Salary from years on the job, with a floor under it -- a beginner does not start
    at zero. The cleanest arena in the collection for seeing what a bias term is for.

    Lesson:       Bias, isolated. Identical in shape to the no-bias income arenas
                  except that the line misses the origin.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5), in thousands, floored at 0
    Solvable by:  A single neuron, but only one with a bias.
    Watch for:    The bias climbing toward 30. Disable it and the weight overshoots
                  trying to compensate.
""",

"Basic_Income_1Input_Thousands.py": """
    Salary from years on the job, counted in thousands.

    Lesson:       Scale, isolated. This is the Dollars generator with every constant
                  divided by 1000 and nothing else touched.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5), floored at 0
    Solvable by:  A single neuron with a bias.
    Watch for:    How much faster this converges than Basic_Income_1Input_Dollars,
                  given that no other thing differs.
""",

"Basic_Income_2Inputs_5Coefficients.py": """
    Salary from experience and years of college, where the two work together --
    a degree is worth more to someone who already has years behind them.

    Lesson:       An interaction term plus a mild quadratic. The two inputs stop
                  being independent.
    Truth:        Salary = 14000 + 10000 x Exp + 8000 x College
                  + 10 x Exp^2 + 2500 x Exp x College
    Solvable by:  One hidden layer, or a single neuron handed Exp x College.
    Watch for:    A linear fit failing worst where both inputs are high.
    Notes:        Noise sigma is 0, so the surface is exact and any error is model error.
""",

"Basic_Income_2Inputs_Dollars.py": """
    Experience, years of college, and salary. College sits on the table as an input
    and contributes nothing -- its coefficient is multiplied away to zero.

    Lesson:       A dead column, in an arena whose name gives no warning about it.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 15). College is never used.
    Solvable by:  A single neuron with a bias.
    Watch for:    The College weight collapsing to zero. If it does not, the model
                  is fitting noise.
    Notes:        Named Dollars but generated in thousands -- the constants are 30 and 4.
                  Trap_Income_DeadInput is the same trap, labelled honestly.
""",

"Basic_Income_2Inputs_Engineered.py": """
    Experience and college, where salary is the product of the two rather than the
    sum -- and the product is handed over as a third column.

    Lesson:       Feature engineering. Multiplying two inputs is out of reach for one
                  neuron until you multiply them yourself.
    Truth:        Salary = (30 + 4 x Exp + noise(sigma 5)) x (College + 0.5)
    Solvable by:  A single neuron, thanks to the Exp x College column.
    Watch for:    Real weight on Exp x College. Drop that column and the same
                  architecture stops working -- see Basic_Income_2Inputs_Multiplicative.
""",

"Basic_Income_2Inputs_Linear.py": """
    Experience, years of college, and salary, with nothing hidden and nothing noisy.
    The one arena where the trained parameters should match the generator exactly.

    Lesson:       Ground truth recovery. You can read the answer off the model and
                  check it against the source.
    Truth:        Salary = 14000 + 12000 x Exp + 8000 x College
    Solvable by:  A single neuron.
    Watch for:    Bias 14000, weights 12000 and 8000. Anything else is an optimizer
                  problem, not a data problem.
    Notes:        Noise sigma is 0, so a perfect fit is genuinely available.
""",

"Basic_Income_2Inputs_Multiplicative.py": """
    Salary that scales with education rather than adding to it -- a year of college
    multiplies what experience already earned. Someone with neither earns very little.

    Lesson:       Multiplicative structure. Additive models miss the high corner
                  in a way that is systematic, not random.
    Truth:        Salary = (30 + 4 x Exp + noise(sigma 5)) x (College + 0.5)
    Solvable by:  One hidden layer.
    Watch for:    Underprediction concentrated where both inputs are high.
    Notes:        Same generator as Basic_Income_2Inputs_Engineered without the
                  pre-computed product column. Run the pair back to back.
""",

"Basic_Income_2Inputs_Nonlinear.py": """
    Salary where early career years are worth far more than late ones, and college
    stops paying after the fourth year. Both effects are step changes, not curves.

    Lesson:       Piecewise coefficients hiding inside what looks linear at a glance.
    Truth:        Base 24000; Exp pays 2500/yr under 10, 1500 under 20, 500 above;
                  College pays 4000/yr to 4 years then 2000; noise sigma 1000
    Solvable by:  One hidden layer.
    Watch for:    Error spiking at experience 10 and 20, where the rate changes.
""",

"Basic_Income_2Inputs_Piecewise.py": """
    A career with three distinct phases -- fast early growth, a slower middle, and a
    plateau. The boundaries between them are hard edges, not gradual transitions.

    Lesson:       Sharp regime changes with no column marking where they happen.
    Truth:        Exp contributes 10000/yr below 10, 5000/yr to 30, then 2000/yr;
                  plus 8000 x College; base 20000; noise sigma 1000
    Solvable by:  One hidden layer, wide enough to place two kinks.
    Watch for:    The elbows at 10 and 30 years. A smooth fit rounds them off.
    Notes:        A rare-spike branch fires at probability 0.000005, which is
                  effectively never. An old comment calls it 5 percent.
""",

"Basic_Income_2Inputs_Quadratic.py": """
    Salary that peaks in mid-career and falls away on both sides, on top of the usual
    climb. Twenty years in is the top of the hill.

    Lesson:       A quadratic term. The single-neuron income arenas end here.
    Truth:        Salary = 14000 + 12000 x Exp + 8000 x College - 100 x (Exp - 20)^2
    Solvable by:  One hidden layer.
    Watch for:    A linear model cutting a straight chord through the hill, wrong at
                  both ends and wrong in the middle.
    Notes:        Noise sigma is 0. A gentler curvature constant is written and then
                  overwritten on the next line -- see the dead code list.
""",

"Basic_Income_2Inputs_Thousands.py": """
    Salary that rises quickly at first and then flattens out, driven by experience
    alone. College is present as a column and contributes nothing.

    Lesson:       A saturating curve and a dead column, neither announced.
    Truth:        Salary = 45 + 35 x (1 - 1/(1 + 0.15 x Exp)) + noise(sigma 8),
                  in thousands. College is multiplied by zero.
    Solvable by:  One hidden layer.
    Watch for:    The College weight going to zero, and the fit flattening above
                  roughly 20 years.
""",

"Basic_Income_3Inputs_Linear.py": """
    Experience, college, and professional certifications against salary. Three
    inputs, all of them real, none of them interacting.

    Lesson:       Scaling up input count without adding difficulty. The step after
                  the two-input linear arena.
    Truth:        Salary = 24000 + 22000 x Exp + 2000 x College
                  + 1500 x Certifications + noise(sigma 2000)
    Solvable by:  A single neuron.
    Watch for:    All three weights recoverable, though Experience dwarfs the others.
""",

"Basic_Iris_TwoClass.py": """
    Two iris species told apart by four flower measurements. This is a synthetic
    stand-in drawn from clean per-species ranges, not the real Fisher measurements.

    Lesson:       The gentlest classification in the collection. The two species
                  barely overlap on any axis.
    Truth:        Setosa and Versicolor drawn from separate uniform ranges on all four
    Solvable by:  A single neuron.
    Watch for:    Petal Length alone nearly separating the classes.
    Notes:        Species is coded 0 for Setosa, 1 for Versicolor. Synthetic -- it
                  imitates the shape of the real data without being it.
""",

"Basic_Linear_3X.py": """
    One input, one output, and a straight line through the origin. The smallest
    thing in the collection that can be trained.

    Lesson:       The first arena to open. One weight to find, and you already know
                  what it should come out to.
    Truth:        y = 3x + noise(sigma 1), x drawn from -10 to 10
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight walking to 3 and stopping there.
""",

"Basic_LiteralData.py": """
    A hand-typed table instead of a generator, for when you need the exact same rows
    every single run. Currently holding four credit-score rows.

    Lesson:       Not a lesson. A scratchpad -- edit the list, get that data.
    Truth:        Whatever is typed in the file. Right now: (25,0) (75,1) (35,0) (65,1)
    Solvable by:  A single neuron, on four rows.
    Watch for:    Nothing. This exists to make a run reproducible, not to be learned from.
    Notes:        Two earlier training_data assignments sit above the live one and are
                  overwritten before use. See the dead code list.
""",

"Basic_Moons.py": """
    Two interleaved crescents on a plane -- the standard picture for "a straight line
    will not do this". Generated by scikit-learn and standardized before you see it.

    Lesson:       A curved boundary you can confirm with your own eyes on a scatter plot.
    Truth:        sklearn make_moons at noise 0.2, then standard-scaled
    Solvable by:  One hidden layer.
    Watch for:    Errors along the seam where the two crescents pass each other.
""",

"Basic_ParityCheck.py": """
    Five bits, and whether an odd number of them are set. Flipping any single bit
    flips the answer, which is exactly what makes it hard.

    Lesson:       XOR generalized. No amount of width in a single layer will do it.
    Truth:        Parity = sum(bits) mod 2, five bits drawn independently
    Solvable by:  One hidden layer, minimum.
    Watch for:    Accuracy stuck at 50 percent without a hidden layer. That is the
                  expected result, not a bug to chase.
""",

"Basic_SolarOutput_Periodic.py": """
    A solar panel's output across a year, high in summer and low in winter, with
    clouds knocking it down at random on about one day in ten.

    Lesson:       A cycle plus one-sided outliers. Cloud days can only subtract.
    Truth:        Output = 50 + 40 x sin(2 pi x Day / 365), minus uniform(0, 60) on
                  10 percent of days, plus noise(sigma 2)
    Solvable by:  One hidden layer, or a single neuron given sin and cos of the day.
    Watch for:    The fit sagging below the true curve. The outliers only pull one way.
""",

"Basic_StockPrice_Indicators.py": """
    Six technical indicators and a resulting share price. Several of the six say
    nearly the same thing, and one does nothing three quarters of the time.

    Lesson:       Overlapping predictors. Weights split unstably between columns
                  that carry the same signal.
    Truth:        Price = 100 + 3 x Mom3 + 1.5 x Mom10, plus a random jolt when
                  Volume Spike exceeds 0.6, a sentiment bonus, volatility-scaled
                  noise, and a plus-or-minus 10 news shock
    Solvable by:  One hidden layer, for the threshold effects.
    Watch for:    The two Momentum weights trading places between runs.
""",

"Basic_TrafficFlow_Weather.py": """
    Cars per minute on a road, given the hour, the weather, and whether anything is
    going on nearby. Day of week arrives pre-encoded as a sine and cosine pair.

    Lesson:       Cyclical encoding. Monday and Sunday are neighbours, and two
                  columns are what makes them neighbours.
    Truth:        Flow = (500 - 50 x sin(2 pi x Hour / 24)) x day factor
                  x weather penalty x event penalty, + noise(sigma 10)
    Solvable by:  One hidden layer -- the factors multiply rather than add.
    Watch for:    Day Sin and Day Cos both carrying weight. Either alone is ambiguous.
""",

"Basic_XOR.py": """
    The four rows that broke the perceptron in 1969. Two bits in, and the answer is
    true when exactly one of them is.

    Lesson:       The reason hidden layers exist. Four rows, and a single neuron
                  cannot fit them.
    Truth:        Result = Input 1 XOR Input 2, all four combinations, no noise
    Solvable by:  One hidden layer with two neurons.
    Watch for:    A single neuron parking at 50 percent and staying there forever.
""",

# ─── REAL ─────────────────────────────────────────────────────────────

"Real_CaliforniaHousing.py": """
    Median house value for California census blocks, from eight facts about the
    neighbourhood. Features arrive already standardized.

    Lesson:       Real data with real correlation between columns, at a scale where
                  nothing has to be fixed before training.
    Source:       sklearn fetch_california_housing, 1990 US Census.
    Solvable by:  One hidden layer.
    Watch for:    Median Income carrying most of the weight, and the target flattening
                  against the census cap at 500,001 dollars.
    Notes:        Inputs are standard-scaled inside the arena, so a scaler on top of
                  that has nothing left to do.
""",

"Real_CaliforniaHousing_USD.py": """
    The same census blocks as the standardized version, priced in actual dollars with
    the eight features left raw. Population runs to the thousands while average
    bedrooms hovers near 1.

    Lesson:       What unscaled real data does to training. This is the arena that
                  makes the case for scalers.
    Source:       sklearn fetch_california_housing, 1990 US Census; target x 100,000.
    Solvable by:  One hidden layer, once the inputs are scaled.
    Watch for:    Divergence without a scaler. Compare Real_CaliforniaHousing, which
                  is this same data pre-standardized.
""",

"Real_Titanic_4Features.py": """
    Who lived and who died, from four things printed on the ticket. Anyone missing
    one of the four is thrown out rather than guessed at.

    Lesson:       Rung one of the Titanic ladder. Every engineered version in this
                  family is scored against this number.
    Source:       seaborn-data titanic.csv, fetched over the network at run time.
    Solvable by:  One hidden layer.
    Watch for:    Sex dominating the weights. If it does not, look upstream.
    Notes:        Dropping incomplete rows costs about 20 percent of the passengers,
                  nearly all of them missing Age. Real_Titanic_Imputed keeps them.
""",

"Real_Titanic_7Features.py": """
    The four ticket facts plus family aboard and port of departure. Nothing is
    encoded or repaired -- the question is whether raw signal alone buys anything.

    Lesson:       Rung two. More columns, no cleverness, and an honest answer about
                  what that gets you.
    Source:       seaborn-data titanic.csv, fetched over the network at run time.
    Solvable by:  One hidden layer.
    Watch for:    Embarked carrying weight despite its 0/1/2 coding implying an order
                  that does not exist. Rung three fixes exactly that.
""",

"Real_Titanic_OneHot.py": """
    Rung two with the port of departure split into three yes/no columns instead of
    one number ranked 0, 1, 2.

    Lesson:       Rung three. Isolates what it costs to pretend a category is a quantity.
    Source:       seaborn-data titanic.csv, fetched over the network at run time.
    Solvable by:  One hidden layer.
    Watch for:    The three port columns taking independent weights. The accuracy gap
                  against Real_Titanic_7Features is the entire lesson.
""",

"Real_Titanic_Imputed.py": """
    Rung three with missing values repaired instead of discarded -- a missing age
    becomes the median for that class and sex, a missing port becomes the commonest.

    Lesson:       Rung four. Isolates the value of imputation with everything else
                  held still.
    Source:       seaborn-data titanic.csv, fetched over the network at run time.
    Solvable by:  One hidden layer.
    Watch for:    About 20 percent more rows than the one-hot version. Whether those
                  extra rows help is the experiment.
""",

"Real_Titanic_EffectCoded.py": """
    Rung four with the port columns switched from one-hot to effect coding, so the
    categories are centered rather than counted against a reference.

    Lesson:       Rung five. One encoding change, everything else identical.
    Source:       seaborn-data titanic.csv, fetched over the network at run time.
    Solvable by:  One hidden layer.
    Watch for:    Smaller and more balanced weights on the two port columns than
                  one-hot produced.
""",

"Real_Titanic_Engineered31.py": """
    Thirty-one columns built out of the passenger list -- titles pulled from names,
    decks from cabin numbers, family size, fare per person, how many people shared a
    ticket. Everything the Kaggle leaderboard reaches for.

    Lesson:       Rung six, and the ceiling of feature engineering on this dataset.
    Source:       datasciencedojo titanic.csv, the Kaggle training set with Name,
                  Ticket and Cabin columns intact.
    Solvable by:  One hidden layer, wide.
    Watch for:    Diminishing returns against Real_Titanic_HighSignal12, which does
                  nearly as well on twelve columns.
""",

"Real_Titanic_HighSignal12.py": """
    Twelve columns kept from the thirty-one, with deck, age bins and the class-sex
    cross dropped as more noise than signal. Title stands in for raw sex.

    Lesson:       Rung seven, and the argument for subtraction. Fewer columns,
                  comparable accuracy.
    Source:       datasciencedojo titanic.csv, the Kaggle training set.
    Solvable by:  One hidden layer.
    Watch for:    Whether accuracy actually drops against the 31-column version.
                  If it does not, that is the point being made.
""",

# ─── SHAPE ────────────────────────────────────────────────────────────

"Shape_AutoNormalize.py": """
    Four inputs spanning five orders of magnitude, two of which feed the answer and
    two of which are decoration. The two that matter are the smallest and largest
    columns in the set.

    Lesson:       Magnitude disparity and irrelevance at once. Tests whether a default
                  scaler picks the right treatment per column rather than per table.
    Truth:        Target Y = 5 x Tiny Feature + 0.001 x Huge Feature + noise(sigma 0.5)
    Solvable by:  A single neuron, once the columns are on comparable scales.
    Watch for:    Whether the two Medium Noise columns are suppressed before or after
                  the scaler runs.
""",

"Shape_CenteredData.py": """
    Three hand-written rows where one column is offset from zero and the other never
    moves. Built to be run at, not learned from.

    Lesson:       Scaler probe. Exposes anything that assumes a column has spread.
    Truth:        (0, 0, 14000), (5, 0, 20000), (10, 0, 26000)
    Solvable by:  Not a learning problem.
    Watch for:    The run completing. Zero Feature is the column that breaks things.
""",

"Shape_Income_MixedMagnitudes.py": """
    Salary from years of experience and the employer's annual revenue. One column
    tops out at 40 and the other at a hundred million.

    Lesson:       What unscaled magnitude disparity does to a gradient, on data that
                  reads like a real problem rather than a probe.
    Truth:        Salary = 30000 + 2000 x Years Exp + 0.0001 x Company Revenue
                  + noise(sigma 5000)
    Solvable by:  A single neuron, with scaling.
    Watch for:    Without a scaler, the Years Exp weight barely moving while Company
                  Revenue swings wildly.
""",

"Shape_MedicalCost_Outliers.py": """
    Annual medical spend from age and BMI, with a dial on the class controlling how
    often a rare expensive procedure lands. The dial is off by default.

    Lesson:       Outlier dosage. The same clean problem, contaminated on demand.
    Truth:        Cost = 3000 + 200 x Age + 350 x BMI + noise(sigma 2500),
                  plus uniform(10000, 30000) at rate outlier_factor
    Solvable by:  A single neuron.
    Watch for:    How MSE moves as outlier_factor rises, against MAE or Huber on the
                  identical data.
    Notes:        outlier_factor defaults to 0.0, so out of the box this is clean.
""",

"Shape_MixedMagnitudes.py": """
    Three hand-written rows, one column near 0.1 and the other near 1000. The minimal
    version of the scale problem.

    Lesson:       Scaler probe. Unscaled, the large column swamps the gradient entirely.
    Truth:        (0.1, 1000, 10000), (0.2, 1200, 12000), (0.3, 1100, 14000)
    Solvable by:  Not a learning problem.
    Watch for:    What each scaler does to the small column. Three rows is few enough
                  to check by hand.
""",

"Shape_NonlinearCompression.py": """
    Four hand-written rows where one column jumps by powers of ten and the other never
    changes. Skew and zero variance sitting side by side.

    Lesson:       Scaler probe. Tests log and power transforms against a dead column
                  in the same pass.
    Truth:        (1, 1, 10000), (10, 1, 11000), (100, 1, 12000), (1000, 1, 13000)
    Solvable by:  Not a learning problem.
    Watch for:    A log transform turning the skewed column into an even ramp while
                  doing nothing at all to the constant.
""",

"Shape_OneGiantOutlier.py": """
    Targets all sitting around 10, and exactly one at 10,000. The input has nothing
    to do with the target at all.

    Lesson:       Loss function comparison. MSE chases the monster, MAE ignores it,
                  Huber splits the difference.
    Truth:        Target Y = 10 + uniform(-1, 1), except one random row set to 10000
    Solvable by:  A single neuron predicting a constant, which is the right answer here.
    Watch for:    Where the prediction settles under each loss. That number is the
                  whole experiment.
""",

"Shape_OneGiantOutlier_Marked.py": """
    The one-giant-outlier data with a second column stating outright which row is the
    monster -- the information a robust loss has to infer, handed over instead.

    Lesson:       Whether an explicit flag beats a robust loss function. Run this
                  against Shape_OneGiantOutlier and compare.
    Truth:        As Shape_OneGiantOutlier, with Outlier Marker set to 2 on the spike
                  and 1 on every other row
    Solvable by:  A single neuron.
    Watch for:    Whether MSE with the marker beats MAE without it.
""",

"Shape_OutlierSensitivity.py": """
    Four hand-written rows -- a clean ramp of three, and one sitting far outside the
    range of the others.

    Lesson:       Scaler probe. Min-Max squeezes the three good rows into a corner;
                  Robust should not.
    Truth:        (5, 1, 100000), (10, 2, 150000), (15, 3, 200000), (90, 8, 800000)
    Solvable by:  Not a learning problem.
    Watch for:    The scaled values of the first three rows. Under Min-Max they
                  collapse on top of each other.
""",

"Shape_SparseInputs.py": """
    Twenty columns, nearly all zero on any given row, and only three of them decide
    the label. Which three is drawn at random when the arena is constructed.

    Lesson:       Finding a handful of live columns in a wide, mostly empty matrix.
    Truth:        Label = 1 when the three secret columns sum above 0, otherwise 0
    Solvable by:  A single neuron.
    Watch for:    Three weights separating cleanly from seventeen that stay near zero.
    Notes:        The signal positions are chosen in __init__, so they can differ
                  between runs even at a fixed seed. See the review list.
""",

"Shape_ZeroVariance.py": """
    Three rows, all identical. Every column holds exactly one value, so nothing
    varies at all.

    Lesson:       Scaler probe. Anything that divides by spread meets a zero here.
    Truth:        Every row is (10, 4, 100000).
    Solvable by:  Not a learning problem.
    Watch for:    The run finishing at all. NaN or inf means a scaler guard has a hole.
""",

# ─── TRAP ─────────────────────────────────────────────────────────────

"Trap_AdversarialNoise.py": """
    A clean straight line with one row in ten replaced by a wild value. There is also
    a second input column that has never mattered.

    Lesson:       Contaminated labels. Ten percent bad rows against a relationship
                  that is otherwise exact.
    Truth:        Label = 2 x X + 3, except a 10 percent chance of uniform(-100, 100)
    Solvable by:  A single neuron, with a robust loss.
    Watch for:    MSE dragged far off the true line by the bad rows, while Y noise
                  holds zero weight throughout.
""",

"Trap_BitFlip_Memory.py": """
    Eight bits in, and the same eight bits back out with one position inverted. Which
    position is supplied as an input, so the task is copy-with-one-change.

    Lesson:       Multi-output. The target here is eight columns wide, not one.
    Truth:        Output bits equal input bits, with the bit at flip_index inverted
    Solvable by:  A single neuron per output bit, given the index is handed over.
    Watch for:    Whether the engine handles a wide target at all. This is as much a
                  plumbing test as a learning one.
    Notes:        The class labels Not Flipped / Flipped do not fit an 8-column target.
                  See the review list.
""",

"Trap_ChaoticSine.py": """
    A wave that oscillates faster the further you get from zero, because its frequency
    depends on x squared. No noise at all -- the difficulty is entirely in the shape.

    Lesson:       A function that is smooth everywhere and still changes character
                  completely across its range.
    Truth:        Target Y = sin(x) + sin(x^2), x drawn from -5 to 5
    Solvable by:  Several hidden layers, and it will still miss the fast ends.
    Watch for:    A good fit near zero and a mess at both extremes, where the
                  oscillation outruns whatever resolution the network has.
""",

"Trap_DeceptiveRegimes.py": """
    Four inputs and a target, where the rule generating it rotates through three
    different forms in blocks of a hundred rows. Which block you are in is never said.

    Lesson:       Conflicting rules with no selector column. A weight that is right
                  in one regime is wrong in the next.
    Truth:        Cycling every 100 rows -- 10x1 + 5x2, then -8x2 + 6x3,
                  then 20x4; noise sigma 1 throughout
    Solvable by:  Not an FFNN. Nothing in the inputs identifies the regime.
    Watch for:    x2 reversing sign between the first two regimes, so its learned
                  weight lands on the average of two opposite truths.
""",

"Trap_DelayedEffect_BloodSugar.py": """
    Meal sizes and blood sugar readings, where a meal shows up in the bloodstream
    three readings later. Every row pairs a meal with a number it did not cause.

    Lesson:       The clearest memory trap in the collection. The cause is in the
                  data, just never on the same row as its effect.
    Truth:        Blood Sugar(t) = 80 + 0.5 x Meal Size(t-3) + noise(sigma 5)
    Solvable by:  Not an FFNN. A row-at-a-time model cannot reach backwards.
    Watch for:    Meal Size weight going to near zero, which is correct -- on its own
                  row it explains nothing.
""",

"Trap_HiddenRegime_Drift.py": """
    Two input signals and a target, where the rule producing it flips every 150 rows
    and then flips back. Time is not one of the columns.

    Lesson:       Non-stationarity. The data looks learnable in any window you pick
                  and is not learnable overall.
    Truth:        Alternating every 150 rows -- 0.5x1 + 0.5x2 + a sine bump, or
                  1.5x1 - 0.7x2 + 20 minus that bump; noise sigma 3
    Solvable by:  Not an FFNN, absent a time column.
    Watch for:    The model settling on the average of two rules and being wrong in
                  both regimes rather than right in either.
""",

"Trap_HiddenSwitch_Power.py": """
    One input, one output, and a coin flipped invisibly on every row deciding which
    of two straight lines you are looking at. The lines slope opposite ways.

    Lesson:       The simplest unlearnable arena here. Two valid answers per input
                  and nothing whatsoever to tell them apart.
    Truth:        Power Usage = 5x + 10 or -3x - 7, chosen at random per row,
                  + noise(sigma 1)
    Solvable by:  Nothing. The switch is not observable.
    Watch for:    A flat line through the middle of two clouds. That is the correct
                  answer to an impossible question.
""",

"Trap_Income_DeadInput.py": """
    Experience, years of college, and salary, where college sits on the table and does
    nothing. The column name says so out loud.

    Lesson:       A dead input, labelled honestly. Confirms the model can zero a
                  column it has no use for.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5). College is never read.
    Solvable by:  A single neuron with a bias.
    Watch for:    The College weight going to zero and staying. Compare
                  Basic_Income_2Inputs_Dollars -- same trap, no warning in the label.
""",

"Trap_Income_LocalMinima.py": """
    Salary built from a linear climb, a squared term, an interaction, and two sine
    waves laid over the top. The sine waves are there to put ripples in the error
    surface for the optimizer to fall into.

    Lesson:       Local minima. Different starting weights land in different places
                  and stay there.
    Truth:        15000 + 9000 Exp + 7000 College + 15 Exp^2 + 1200 Exp x College
                  + 3000 sin(Exp/2) + 1000 sin(1.5 College)
    Solvable by:  Two hidden layers.
    Watch for:    Final loss varying widely across seeds with everything else fixed.
    Notes:        Noise sigma is 0, so run-to-run difference comes entirely from
                  initialization rather than from the data.
""",

"Trap_MarketSentiment.py": """
    Four economic indicators against a consumer confidence number, day by day across
    several years. The indicators drift with the seasons and one reverses polarity
    twice a year.

    Lesson:       Inputs that look chaotic feeding a target that is not. Redundant,
                  noisy and phase-shifted all at once.
    Truth:        Consumer Confidence = 0.3 Volatility + 0.2 Sentiment
                  - 0.025 Unemployment Claims + 0.25 Survey Skew + noise(sigma 2)
    Solvable by:  A single neuron -- the blend is linear despite the drift.
    Watch for:    How much easier this turns out to be than the inputs suggest.
""",

"Trap_MultiModal_Interactions.py": """
    Six inputs feeding a target assembled one way above a threshold and an entirely
    different way below it, then scaled by a context term built from two other columns.

    Lesson:       Bimodal output with threshold effects and noise that scales with
                  the signal rather than sitting at a fixed level.
    Truth:        Two regimes chosen by Mode Switcher, each with its own polynomial,
                  plus a high-order interaction active only in a narrow region
    Solvable by:  Four or more hidden layers, per the author's own note.
    Watch for:    A histogram of the target showing two humps rather than one.
    Notes:        Carries a get_problem_characteristics method nothing calls.
""",

"Trap_MultiModal_Temperature.py": """
    Temperature readings drawn from two separate populations, one clustered near 20
    degrees and one near 28. Hour of day is the only input and it tells you nothing.

    Lesson:       A two-humped target. The mean sits in the gap between the humps,
                  at a value that essentially never occurs.
    Truth:        Temperature = 20 + noise(sigma 1) or 28 + noise(sigma 2), chosen at
                  random; Hour of Day is independent of both
    Solvable by:  Nothing. The mode is not observable.
    Watch for:    The prediction landing near 24 and the loss refusing to improve.
""",

"Trap_NestedSineFlip.py": """
    A wave that turns upside down for part of its range and then rights itself. Where
    you are on the axis is an input, so unlike most of its neighbours in this family
    this one is genuinely learnable.

    Lesson:       Architecture limits rather than impossibility. Watch a network that
                  is too small fail on purpose.
    Truth:        Signal = sin(3 x Position), negated for 3 <= Position < 6,
                  + noise(sigma 0.1)
    Solvable by:  Hidden layers, wider than you would guess.
    Watch for:    Error piling up at the two reversal points while the outer bands
                  fit cleanly.
""",

"Trap_PathologicalChaos.py": """
    Eight inputs feeding a target assembled from a logistic map, a fractal escape
    count, a logarithmic singularity, an inverse-square term and Cauchy noise. Built
    to be unfittable, as a control.

    Lesson:       The far end of the difficulty scale. Useful mainly for showing what
                  a genuinely hopeless loss curve looks like.
    Truth:        Layered chaotic, fractal and discontinuous terms under heavy-tailed
                  noise; read the generator, it does not summarize
    Solvable by:  Nothing on offer. The author suggests eight-plus layers and stops
                  short of promising it helps.
    Watch for:    Loss that does not descend. Decoy Correlation is a deliberate plant.
    Notes:        Cauchy noise has no finite variance, so loss numbers here are not
                  comparable to any other arena. Carries a dead
                  get_problem_characteristics method.
""",

"Trap_PiecewiseRegime.py": """
    Two inputs and a target that follows one rule on the left half of the range and a
    different one on the right. The two rules disagree about the sign of X.

    Lesson:       A conditional split where the condition is visible in the data.
                  Learnable, unlike the hidden-regime arenas next door.
    Truth:        Label = 3x + 2y + 5 when x < 10, otherwise -2x + 0.5y - 10
    Solvable by:  One hidden layer.
    Watch for:    A single neuron fitting a compromise slope that is wrong on both sides.
""",

"Trap_RedHerringFeatures.py": """
    One input that determines the answer exactly, and two that are pure noise ranging
    twenty times wider. The target itself has no noise at all.

    Lesson:       Ignoring loud, irrelevant columns. The distractors dwarf the real
                  signal in magnitude.
    Truth:        Label = 4 x Signal - 7, exactly. Noise1 and Noise2 are never read.
    Solvable by:  A single neuron.
    Watch for:    Both noise weights driving to zero. A perfect fit is available here,
                  so anything short of one is distraction.
""",

"Trap_RedundantFeatures.py": """
    Two inputs that are almost the same number -- one is the other with a small jitter
    added -- and only the original drives the target.

    Lesson:       Collinearity. Two columns carrying the same information have no
                  unique correct weighting between them.
    Truth:        Redundant Copy = Signal + uniform(-0.2, 0.2);
                  Label = 2 x Signal + 3, exactly
    Solvable by:  A single neuron.
    Watch for:    The two weights summing to about 2 while splitting between
                  themselves differently on every run.
""",

"Trap_RegimeTrigger_Memory.py": """
    Two inputs that look identical in either regime, and a target whose rule toggles
    every seventeen rows. The toggle lives in the generator and never becomes a column.

    Lesson:       Proof that a stateless network has a ceiling. Two contradictory
                  answers per input and no way to tell which one applies.
    Truth:        Target = 2x1 + 3x2 or -2x1 + 1.5x2, alternating every 17 rows,
                  + noise(sigma 0.2)
    Solvable by:  Not an FFNN. This arena is here to fail.
    Watch for:    The learned weights averaging the two rules -- near zero on x1,
                  near 2.25 on x2.
""",

"Trap_TargetDrift_Commodity.py": """
    A price that depends on one input plus a baseline creeping upward forever. Time is
    given as a column, which is what makes this drift catchable.

    Lesson:       Non-stationary target with the clock supplied. Compare against the
                  drift arenas that hide it.
    Truth:        Price = 100 + cumulative drift + 2 x Demand + noise(sigma 2);
                  drift adds uniform(0, 0.1) per row
    Solvable by:  A single neuron, using Time.
    Watch for:    A positive weight on Time. Remove that column and the model is lost.
""",
}


# ══════════════════════════════════════════════════════════════════════
# RENAMES — applied to return / labels lists only
# ══════════════════════════════════════════════════════════════════════

TITANIC = [("Pclass", "Ticket Class"), ("Outcome", "Survival")]

ArenaDocPatch.RENAMES = {
    "Basic_CarValue_2Inputs.py"      : [("Value", "Car Value")],
    "Basic_Iris_TwoClass.py"         : [("Species (0=Setosa, 1=Versicolor)", "Species")],
    "Basic_Moons.py"                 : [("Answer", "Moon")],
    "Real_CaliforniaHousing.py"      : [("MedianHouseValue", "Median House Value")],
    "Real_Titanic_4Features.py"      : TITANIC,
    "Real_Titanic_7Features.py"      : TITANIC,
    "Real_Titanic_OneHot.py"         : TITANIC,
    "Real_Titanic_Imputed.py"        : TITANIC,
    "Real_Titanic_EffectCoded.py"    : TITANIC,
    "Real_Titanic_Engineered31.py"   : TITANIC,
    "Real_Titanic_HighSignal12.py"   : TITANIC,
    "Trap_NestedSineFlip.py"         : [("x", "Position")],
    "Trap_TargetDrift_Commodity.py"  : [("Feature", "Demand")],
}


if __name__ == "__main__": ArenaDocPatch.run()


# ══════════════════════════════════════════════════════════════════════
# LIST 1 — FIELD RENAMES
# ══════════════════════════════════════════════════════════════════════
#   Basic_CarValue_2Inputs.py      : Value                -> Car Value
#   Basic_Iris_TwoClass.py         : Species (0=Setosa..) -> Species
#   Basic_Moons.py                 : Answer               -> Moon
#   Real_CaliforniaHousing.py      : MedianHouseValue     -> Median House Value
#   Real_Titanic_*.py  (7 files)   : Pclass               -> Ticket Class
#   Real_Titanic_*.py  (7 files)   : Outcome              -> Survival
#   Trap_NestedSineFlip.py         : x                    -> Position
#   Trap_TargetDrift_Commodity.py  : Feature              -> Demand
#
#   DELIBERATELY LEFT ALONE — the abstract arenas
#     Input X / Target Y / Label / Signal / Target / x1 / x2 stay as they are, in
#     Basic_Linear_3X, Trap_ChaoticSine, Shape_OneGiantOutlier (+ _Marked),
#     Shape_AutoNormalize, Shape_SparseInputs, Trap_AdversarialNoise,
#     Trap_PiecewiseRegime, Trap_RegimeTrigger_Memory, Trap_DeceptiveRegimes,
#     Trap_HiddenRegime_Drift.
#     These arenas are about shape, not subject. Inventing "Position" or "Demand"
#     for a pure math probe would be fiction, and Goal: Predict Target Y is at
#     least honest. Say the word and I will name them.
#
#   OUT OF REACH — generated at run time, cannot be renamed by string replace
#     Real_CaliforniaHousing   8 input names come from sklearn feature_names
#     Basic_ParityCheck        Bit 1..Bit 5 built by f-string
#     Shape_SparseInputs       X1..X20 built by f-string
#     Trap_BitFlip_Memory      bit_0..bit_7 and bit_N_flipped built by f-string
#                              (these are also the only lowercase labels in the set)

# ══════════════════════════════════════════════════════════════════════
# LIST 2 — UNSURE, CHECK THESE FIRST
# ══════════════════════════════════════════════════════════════════════
#   Basic_CustomerChurn_3Inputs   THE BIG ONE. churn_probability is never clamped.
#     Monthly Spend alone contributes up to 5.0, so P(churn) exceeds 1 for the
#     large majority of rows and nearly everything churns. I documented it as an
#     imbalance exercise because that is what it currently is, but I suspect the
#     intent was 0.005 not 0.05. Your call -- I did not touch the generator.
#
#   Basic_Income_2Inputs_Dollars  Named Dollars, generated in thousands (30 and 4,
#     not 30000 and 4000). Also its College column is multiplied by zero, making it
#     an unlabelled twin of Trap_Income_DeadInput. Both facts are now in the
#     docstring, but the filename still lies.
#
#   Trap_BitFlip_Memory   Returns 17 columns where the last 8 are the target, plus
#     class labels ["Not Flipped", "Flipped"] that describe a single binary output.
#     Those two facts cannot both be right. The docstring describes what the code
#     does, not what the class labels claim.
#
#   Shape_SparseInputs   signal_indices is drawn in __init__ via random.sample.
#     If the seed is set after construction, two runs at the same seed get
#     different signal columns. Contradicts "same seed, same data" on the Arenas tab.
#
#   Trap_DeceptiveRegimes   The old docstring called x4 "one faint, reliable signal".
#     It is not reliable -- x4 only appears in regime 2 and is noise in the other
#     two. I wrote x2 as the interesting column instead, since it actually reverses
#     sign. The field label "x4 (weak true)" still makes the old claim.
#
#   Basic_Baseball_Hits_Matrix   NOT PATCHED. Its own module docstring marks it
#     ATTIC and says it was moved out of coliseum/arenas. Not in DOCS. If it is
#     still in your folder, tell me and I will write one.
#
#   Titanic ladder rung numbering   I asserted the order 4Features, 7Features,
#     OneHot, Imputed, EffectCoded, Engineered31, HighSignal12 from the existing
#     docstrings and the V6/V7/V8 comments. EffectCoded's old docstring says
#     "Version 6" but it is the fifth file in that sequence. Verify before this
#     ships as teaching text.
#
#   Numbers I computed rather than read
#     CreditScore_Repaid   75% ceiling. Best rule is repaid-if-score>50; expected
#                          accuracy over a uniform 1..100 score is exactly 0.750.
#     Titanic 4Features    ~20% row loss. 177 of 891 seaborn rows lack Age.
#     Turbine              0.000306 = 0.5 x 1.225 x 0.0005.
#     Quadratic            -100 x (Exp-20)^2 = the surviving -40000 x (...)^2 / 400.
#     RegimeTrigger        x2 average of 3 and 1.5 = 2.25.

# ══════════════════════════════════════════════════════════════════════
# LIST 3 — DEAD CODE AND BUGS SPOTTED, NOT TOUCHED
# ══════════════════════════════════════════════════════════════════════
#   Basic_LiteralData.py          WORST OFFENDER. Eight commented-out return
#     statements, then three consecutive training_data assignments where only the
#     third survives. The first two (a 10-row income table and a 4-row XOR-ish
#     table) are written and immediately overwritten. Live data is 4 credit rows.
#
#   Basic_Income_1Input_Thousands.py   college = random.uniform(0, 8)  never read
#   Basic_Income_1Input_Dollars.py     same three lines -- file is a copy
#   Trap_Income_DeadInput.py           same college line, plus a commented append
#
#   Basic_Income_2Inputs_Quadratic.py
#     nonlinear_component assigned twice; the -100 x (...) line is overwritten by
#     the -40000 x (...) / 400 line before use.
#
#   Basic_Income_2Inputs_Dollars.py
#     edu_contribution assigned twice; the diminishing-returns formula is
#     overwritten by  0 * salary * college  on the next line.
#
#   Basic_Income_2Inputs_Thousands.py
#     edu_contribution = 0 * (...)  -- the multiplier was zeroed, not removed.
#     Commented original sits directly above.
#
#   Zero-sigma noise, written as if live
#     Basic_Income_2Inputs_Linear         random.gauss(0, 0000)
#     Basic_Income_2Inputs_5Coefficients  random.gauss(0, 0)
#     Basic_Income_2Inputs_Quadratic      random.gauss(0, 0)
#     Trap_Income_LocalMinima             random.gauss(0, 0)
#
#   Basic_Income_2Inputs_Piecewise.py
#     if random.random() < 0.000005  with the comment  # 5% chance.
#     Off by four orders of magnitude. The spike branch is effectively dead.
#
#   Basic_HouseValue_1Input.py
#     Docstring said 500-4000 sq ft; code uses 1500-4000. Three commented price
#     experiments below it.
#
#   Basic_Baseball_Hits_Flat.py
#     return training_data, [  # NEW - dedent this whole block 8 spaces
#     A leftover merge instruction living in the source.
#
#   Trap_MultiModal_Interactions.py   get_problem_characteristics  zero callers
#   Trap_PathologicalChaos.py         get_problem_characteristics  zero callers
#     (Only the retired duplicates MultiModal_Nonlinear_Interactions.py and
#      Pathological_Discontinuous_Chaos.py also define it. Nothing calls either.)
#
#   Trap_RegimeTrigger_Memory.py
#     self._regime_flag  -- underscore prefix, against house convention.
#
#   Basic_CreditScore_Repaid.py
#     return training_data    , [...]   stray spaces before the comma. Cosmetic,
#     and it sits on a line the rename pass walks, so I left it alone.
#
#   Trailing preview comments, harmless but noisy
#     Trap_DeceptiveRegimes, Trap_HiddenRegime_Drift, Trap_MarketSentiment,
#     Trap_RegimeTrigger_Memory  -- each ends with a commented "preview" snippet
#     referencing class names that no longer exist.