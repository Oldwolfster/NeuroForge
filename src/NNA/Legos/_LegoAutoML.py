from dataclasses import asdict
from typing import Any

class LegoAutoML:
    def __init__(self):
        self.applied_rules = []
        self.seen_prints = set()
        self.applied_pairs = set()

        print(f"\t🧠🧠 Welcome to 'Smart Configuration' Anything not set in your model will be set to optimal conditions(hopefully) 🧠🧠")
        print(f"\t🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠")
        print(f"\t🧠🧠 Settings Set:🧠🧠")
        print(f"\t🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠")

    def maybe_print(self, text_to_print_once: str) -> None:
        if text_to_print_once not in self.seen_prints:
            print(text_to_print_once)
            self.seen_prints.add(text_to_print_once)

    def apply(self, config, rules, tri):
        # track which rule‐indices we’ve applied (and to what value)
        # keep running until no rule makes an actual change
        while self.apply_single_rule(config, rules, tri):
            pass

    def apply_single_rule(self, config, rules, tri) -> bool:
        """
        Returns True if it applied one rule (and mutated config).
        """
        sorted_rules = sorted(rules, key=lambda rule: rule[1])

        for idx, (override, priority, action, condition) in enumerate(sorted_rules):
            try:
                condition_result = self._safe_eval(condition, config)

                if condition_result:
                    for key, value in action.items():
                        current = getattr(config, key)
                        should_apply = (
                                (override and current != value) or
                                (not override and current is None)
                        )

                        # Skip output if we've already printed the 🧠 for this pair
                        if (priority, key) not in self.applied_pairs:
                            if should_apply:
                                self.maybe_print(f"\t🧠{priority}: {key} = {value}")
                                self.applied_pairs.add((priority, key))  # Mark as applied
                            else:
                                self.maybe_print(
                                    f"\t⏭️{priority}: {key} SKIPPED (current={current}, override={override})")

                        if should_apply:
                            setattr(config, key, value)
                            self.applied_rules.append({
                                "priority": priority,
                                "setting": key,
                                "value": value,
                                "condition": condition
                            })
                            return True
                else:
                    # Condition was False - show why

                    for key in action.keys():
                        # Skip output if we've already applied this pair
                        if (priority, key) not in self.applied_pairs:
                            self.maybe_print(f"\t❌{priority}: {key} BLOCKED (condition False: '{condition}')")

            except Exception as e:
                print(f"⚠️ Rule failed: {priority} {condition} -> {e}")

        return False

    def _safe_eval(self, condition: str, config: object) -> bool:
        tokens = condition.strip().split()
        if not tokens:
            return False

        root_var = tokens[0].split(".")[0]
        if root_var == "not" and len(tokens) > 1:   root_var = tokens[1].split(".")[0]  #Handle 'Not' keyword

        # If the root var looks like a literal (e.g., '1', '"text"', 'True'), go ahead
        if root_var.isdigit() or root_var in {"True", "False"} or root_var.startswith(("'", '"')):
            return bool(eval(condition, {}, config.__dict__))

        # Otherwise, check if the referenced attribute exists
        if getattr(config, root_var, None) is None:
            return False

        return bool(eval(condition, {}, config.__dict__))



