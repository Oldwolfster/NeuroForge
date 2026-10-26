from dataclasses import asdict
from typing import Any



class LegoAutoML:
    def __init__(self, ok_to_print,conn=None, batch_id=None, run_id=None):
        self.applied_rules  = []
        self.seen_prints    = set()
        self.applied_pairs  = set()

        self.ok_to_print    = ok_to_print
        self.conn           = conn
        self.batch_id       = batch_id
        self.run_id         = run_id
        self.seq            = 0

        if (self.ok_to_print):
            print(f"\t🧠🧠 Welcome to 'Smart Configuration' Anything not set in your model will be set to optimal conditions(hopefully) 🧠🧠")
            print(f"\t🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠")
            print(f"\t🧠🧠 Settings Set:🧠🧠")
            print(f"\t🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠")

    def maybe_print(self, text_to_print_once: str) -> None:
        if not self.ok_to_print: return
        if text_to_print_once not in self.seen_prints:
            print(text_to_print_once)
            self.seen_prints.add(text_to_print_once)

    def apply(self, config, rules):
        # track which rule‐indices we’ve applied (and to what value)
        # keep running until no rule makes an actual change
        self.automl_log_clear()
        while self.apply_single_rule(config, rules):
            pass


    def apply_single_rule(self, config, rules) -> bool:
        sorted_rules = sorted(rules, key=lambda rule: rule[1])

        for idx, (override, priority, action, condition) in enumerate(sorted_rules):
            try:
                condition_result = self.safe_eval(condition, config)

                if condition_result:
                    for key, value in action.items():
                        current = getattr(config, key)
                        should_apply = (
                                (override and current != value) or
                                (not override and current is None)
                        )

                        if (priority, key) not in self.applied_pairs:
                            if should_apply:
                                msg = f"\t🧠{priority}: {key} = {value}"
                                self.maybe_print(msg)
                                self.applied_pairs.add((priority, key))
                                self.automl_log_step(priority, key, str(value), 1, override, condition, msg)
                            else:
                                msg = f"\t⏭️{priority}: {key} SKIPPED (current={current}, override={override})"
                                self.maybe_print(msg)
                                self.automl_log_step(priority, key, str(value), 0, override, condition, msg)

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
                    for key in action.keys():
                        if (priority, key) not in self.applied_pairs:
                            msg = f"\t❌{priority}: {key} BLOCKED (condition False: '{condition}')"
                            self.maybe_print(msg)
                            self.automl_log_step(priority, key, "", 0, override, condition, msg)

            except Exception as e:
                print(f"⚠️ Rule failed: {priority} {condition} -> {e}")

        return False

    def safe_eval(self, condition: str, config: object) -> bool:
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


    def automl_log_clear(self):
        if not self.conn: return
        self.conn.execute(
            "DELETE FROM log_automl WHERE batch_id = ? AND run_id = ?",
            (self.batch_id, self.run_id)
        )
        self.conn.commit()

    def automl_log_step(self, priority, setting, value, applied, override, condition, message):
        if not self.conn: return
        self.seq += 1
        self.conn.execute(
            "INSERT INTO log_automl (batch_id, run_id, seq, priority, setting, value, applied, override, condition, message) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.batch_id, self.run_id, self.seq, str(priority), setting, value, int(applied), int(override), condition, message)
        )
        self.conn.commit()
