## Adapted from https://github.com/maxwbuckley/r2ltokenizing/blob/main/llmarithmetic.py; with modifications
import random
import yaml
from typing import Dict, Iterator
from datasets import Dataset, DatasetDict

REPO_ID = "garrethlee/arithmetic_problems"
CONFIG_PATH = "config/default_config.yaml"

class ArithmeticProblemsConfig:
    def __init__(
        self,
        config_file_path: str = CONFIG_PATH,
    ) -> None:
        with open(config_file_path) as f:
            self.config = yaml.safe_load(f)

class ArithmeticProblemGenerator:
    OPERATORS = ["+", "-", "*", "/"]

    def __init__(self, config: ArithmeticProblemsConfig):
        self._config = config.config
        self._dataset_config = self._config.get("dataset_config")
        self._difficulties_config = self._config.get("difficulties")

        self.max_rounding_precision = self._dataset_config.get("max_rounding_precision")
        self.use_commas = self._dataset_config.get("use_commas")
        self.include_decimals = self._dataset_config.get("include_decimals")

    def _generate_number(
        self, min_val: int, max_val: int, is_decimal: bool
    ) -> float | int:
        """
        Generates a random number within a specified range, either as an integer or float.

        Args:
        min_val: The minimum value of the range.
        max_val: The maximum value of the range.
        is_decimal: Whether to generate a float or an int.

        Returns:
        A random number within the specified range, either as an int or a float.
        """
        if is_decimal:
            return random.uniform(min_val, max_val)
        else:
            return random.randint(min_val, max_val)

    def _format_number(self, number: int | float, use_commas: bool = False) -> str:
        """
        Rounds a number to a specified precision, and then formats it as a string.

        Args:
        number: The number to be formatted.
        use_commas: Whether to include commas as thousand separators.

        Returns:
        A string representation of the input number, rounded to the specified precision.
        """
        formatted_number = round(number, self.max_rounding_precision)
        if use_commas:
            return "{:,}".format(formatted_number)
        else:
            return str(formatted_number)

    def _construct_equation(
        self, a: int, b: int, operator="+", commas: bool = False
    ) -> str:
        """Helper function for constructing the string equations."""
        return "%s %s %s = " % (
            self._format_number(a, commas),
            operator,
            self._format_number(b, commas),
        )

    def create_question_answer_pair(
        self, min_value: int, max_value: int, operator: str, is_decimal: bool
    ) -> dict[str, str]:
        """Creates a random question and correct answer pair.

        Args:
        min_value: The lowest possible random value.
        max_value: The highest possible random value.
        include_decimals: Whether to include decimal numbers in the generated problems.
        operator: The mathematical operator to use.
        use_commas: Whether to use commas to separate numbers right-to-left.

        Returns:
        A dictionary containing the equation string and the expected answer.
        """
        if operator not in self.OPERATORS:
            raise ValueError(f"Invalid operator: {operator}")

        operand1, operand2 = [
            self._generate_number(
                min_val=min_value, max_val=max_value, is_decimal=is_decimal
            )
            for _ in range(2)
        ]

        if operator == "-":
            result = operand1 - operand2
        elif operator == "+":
            result = operand1 + operand2
        else:
            result = operand1 * operand2

        if operator == "/":
            # Use operand1 as the dividend so that the result has nicer numbers
            return {
                "question": self._construct_equation(
                    result, operand2, operator, self.use_commas
                ),
                "answer": self._format_number(operand1, self.use_commas),
            }

        return {
            "question": self._construct_equation(
                operand1, operand2, operator, self.use_commas
            ),
            "answer": self._format_number(result, self.use_commas),
        }

    def create_dataset(self) -> DatasetDict:
        """
        Generates the arithmetic problems dataset.

        Returns:
            A DatasetDict containing the generated problems, split by difficulty.
        """

        def generate_problems(
            num_problems: int,
            lower_bound: int,
            upper_bound: int,
            decimal_problem_proportion: float,
        ) -> Iterator[Dict[str, str]]:
            """
            Creates a generator that yields a sequence of dictionaries containing the equation string and expected answer.

            Args:
            num_problems: The number of problems to generate.
            lower_bound: The lower bound of the range of possible values.
            upper_bound: The upper bound of the range of possible values.
            decimal_problem_proportion: The proportion of problems that are decimal.
            """
            for i in range(num_problems):
                yield self.create_question_answer_pair(
                    min_value=lower_bound,
                    max_value=upper_bound,
                    operator=random.choice(self.OPERATORS),
                    is_decimal=i < int(num_problems * decimal_problem_proportion),
                )

        def generate_split(shuffle: bool = True, **kwargs) -> Dataset:
            """
            Generates a split of the arithmetic problems dataset.

            Args:
            **kwargs: lower_bound and upper_bound to pass to generate_problems.

            Returns:
            A Dataset containing the generated problems.
            """

            lower_bound = 10 ** kwargs.get("min_exponent")
            upper_bound = 10 ** kwargs.get("max_exponent")

            split = Dataset.from_generator(
                generate_problems,
                gen_kwargs={
                    "num_problems": kwargs.get("num_problems"),
                    "decimal_problem_proportion": kwargs.get(
                        "decimal_problem_proportion"
                    ),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                },
            )

            if shuffle:
                split = split.shuffle()

            return split

        dataset_dict = {}

        for difficulty, difficulty_config in self._difficulties_config.items():
            dataset_dict[difficulty] = generate_split(shuffle=True, **difficulty_config)

        return DatasetDict(dataset_dict)


if __name__ == "__main__":
    dataset_config = ArithmeticProblemsConfig()
    generator = ArithmeticProblemGenerator(config=dataset_config)
    generated_data = generator.create_dataset()
    generated_data.push_to_hub(repo_id=REPO_ID)
