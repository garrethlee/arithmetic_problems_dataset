## Adapted from https://github.com/maxwbuckley/r2ltokenizing/blob/main/llmarithmetic.py; with modifications
import random
import yaml
from typing import Dict, Iterator
from datasets import Dataset, DatasetDict
from pathlib import Path

REPO_ID = "garrethlee/arithmetic_problems"
CONFIG_PATH = (
    Path(__file__).parent.absolute() / "config" / "default_config.yaml"
).as_posix()


class Operator:
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"

    OPERATORS = [ADD, SUBTRACT, MULTIPLY, DIVIDE]

    @classmethod
    def is_operator(cls, value):
        return value in cls.OPERATORS


class OperationType:
    INT_INT = [False, False]
    INT_FLOAT = [True, False]
    FLOAT_FLOAT = [True, True]


class ArithmeticProblemsConfig:
    def __init__(
        self,
        config_file_path: str = CONFIG_PATH,
    ) -> None:
        with open(config_file_path) as f:
            self.config = yaml.safe_load(f)


class ArithmeticProblemGenerator:
    FLOAT_ANSWER_ROUNDING_PRECISION = 4

    def __init__(self, config: ArithmeticProblemsConfig):
        self._config = config.config
        self._dataset_config = self._config.get("dataset_config")
        self._split_config = self._config.get("splits")

        self.use_commas = self._dataset_config.get("use_commas")
        self.float_float_problem_proportion = self._dataset_config.get(
            "float_float_problem_proportion"
        )
        self.test_size = self._dataset_config.get("test_size")

    def _generate_number(
        self, min_val: int, max_val: int, is_float: bool, max_rounding_precision: int
    ) -> float | int:
        """
        Generates a random number within a specified range, either as an integer or float.

        Args:
        min_val: The minimum value of the range.
        max_val: The maximum value of the range.
        is_float: If true, generates a float
        max_rounding_precision: The maximum precision to use when rounding the number.

        Returns:
        A random number within the specified range, either as an int or a float.
        """
        if is_float:
            # Round to a random precision between 0 and max_rounding_precision
            return round(
                random.uniform(min_val, max_val),
                random.choice(range(1, max_rounding_precision+1)),
            )
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
        if use_commas:
            return "{:,}".format(number)
        else:
            return str(number)

    def _construct_equation(
        self,
        operand1: int | float,
        operand2: int | float,
        operator: str,
        use_commas: bool = False,
    ) -> str:
        """Helper function for constructing the string equations."""

        if random.random() < 0.5 and operator != Operator.DIVIDE:
            operand1, operand2 = operand2, operand1

        return "%s %s %s = " % (
            self._format_number(operand1, use_commas),
            operator,
            self._format_number(operand2, use_commas),
        )

    def create_question_answer_pair(
        self,
        min_value: int,
        max_value: int,
        operator: str,
        operation_type: list[bool],
        max_rounding_precision: int | None,
    ) -> dict[str, str]:
        """Creates a random question and correct answer pair.

        Args:
        min_value: The lowest possible random value.
        max_value: The highest possible random value.
        include_decimals: Whether to include float numbers in the generated problems.
        operator: The mathematical operator to use.
        use_commas: Whether to use commas to separate numbers right-to-left.

        Returns:
        A dictionary containing the equation string and the expected answer.
        """
        if not Operator.is_operator(operator):
            raise ValueError(f"Invalid operator: {operator}")

        is_float1, is_float2 = operation_type
        operand1 = self._generate_number(
            min_val=min_value,
            max_val=max_value,
            is_float=is_float1,
            max_rounding_precision=max_rounding_precision,
        )
        operand2 = self._generate_number(
            min_val=min_value,
            max_val=max_value,
            is_float=is_float2,
            max_rounding_precision=max_rounding_precision,
        )

        if operator == Operator.SUBTRACT:
            result = operand1 - operand2
        elif operator == Operator.ADD:
            result = operand1 + operand2
        elif operator == Operator.MULTIPLY:
            result = operand1 * operand2
        else:
            operand2 = max(1, operand2)
            tmp = operand1 / operand2

            if operation_type == OperationType.INT_INT:
                # prevents zero division
                operand1 = int(round(tmp)) * operand2
                result = int(operand1 / operand2)

            elif operation_type == OperationType.INT_FLOAT:
                operand2 = max(10 ** (-max_rounding_precision), operand2)
                result = tmp

            else:
                operand2 = max(10 ** (-max_rounding_precision), operand2)
                result = tmp

        result = round(result, self.FLOAT_ANSWER_ROUNDING_PRECISION)

        question = self._construct_equation(
            operand1=operand1,
            operand2=operand2,
            operator=operator,
            use_commas=self.use_commas,
        )
        answer = self._format_number(result, self.use_commas)

        return {"question": question, "answer": answer}

    def create_dataset(self) -> DatasetDict:
        """
        Generates the arithmetic problems dataset.

        Returns:
            A DatasetDict containing the generated problems, split by split.
        """

        def generate_problems(
            num_problems: int,
            lower_bound: int,
            upper_bound: int,
            max_rounding_precision: int | None,
        ) -> Iterator[Dict[str, str]]:
            def _get_operation_type(num_problems_per_operator: int, current_idx: int):
                # If max_rounding_precision is None, generate only integer problems
                """
                Determines the type of operation (integer-integer, float-float, or integer-float)
                to generate based on the current index and the proportion of float problems.

                Args:
                    current_idx: The current index of the problem being generated.
                    num_problems: The total number of problems to generate.
                    max_rounding_precision: The maximum rounding precision to use when generating float problems.

                Returns:
                    An OperationType indicating the type of operation to generate.
                """
                if max_rounding_precision is None:
                    return OperationType.INT_INT

                # Otherwise, if the current index is less than the float problem proportion,
                elif (
                    current_idx
                    < num_problems_per_operator * self.float_float_problem_proportion
                ):
                    return OperationType.FLOAT_FLOAT

                else:
                    return OperationType.INT_FLOAT

            for operator in Operator.OPERATORS:
                num_problems_per_operator = num_problems // 4
                # Generate questions for each +, -, * , and /
                for i in range(num_problems_per_operator):
                    yield self.create_question_answer_pair(
                        min_value=lower_bound,
                        max_value=upper_bound,
                        operator=operator,
                        operation_type=_get_operation_type(
                            num_problems_per_operator, i
                        ),
                        max_rounding_precision=max_rounding_precision,
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
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "max_rounding_precision": kwargs.get(
                        "max_rounding_precision", None
                    ),
                },
            )

            if shuffle:
                split = split.shuffle()

            return split

        dataset_dict = {}

        for split, split_config in self._split_config.items():
            dataset_dict[split] = generate_split(
                shuffle=True, **split_config
            ).train_test_split(test_size=self._dataset_config.get("test_size"))
            dataset_dict[f"{split}_train"] = dataset_dict[split].pop("train")
            dataset_dict[f"{split}_test"] = dataset_dict[split].pop("test")
            del dataset_dict[split]

        return DatasetDict(dataset_dict)


if __name__ == "__main__":
    dataset_config = ArithmeticProblemsConfig()
    generator = ArithmeticProblemGenerator(config=dataset_config)
    generated_data = generator.create_dataset()
    generated_data.push_to_hub(repo_id=REPO_ID)
