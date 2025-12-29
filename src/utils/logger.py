from _typeshed import SupportsWrite
from sys import stdout


class Style:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    RESET = "\033[0m"

    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    MAGENTA_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    INVERT = "\033[7m"


class Logger:
    def __init__(self, fd: SupportsWrite[str] = stdout) -> None:
        self.fd = fd

    def log(self, *messages: str, color: str = Style.GREEN, sep: str = "\n") -> None:
        print(f"{color}{sep.join(messages)}{Style.RESET}", file=self.fd)

    def warn(self, *messages: str, sep: str = "\n") -> None:
        print(f"{Style.YELLOW}{Style.BOLD}Warning: {sep.join(messages)}{Style.RESET}", file=self.fd)

    def error(self, *messages: str, sep: str = "\n") -> None:
        print(f"{Style.RED}{Style.INVERT}Error: {sep.join(messages)}{Style.RESET}", file=self.fd)

    def dict_log(
        self,
        data: dict,
        key_color: str = Style.CYAN,
        value_color: str = Style.YELLOW,
        sep: str = "\n",
    ) -> None:
        print(
            *(
                f"{key_color}{key}: {value_color}{value}{Style.RESET}"
                for key, value in data.items()
            ),
            file=self.fd,
            sep=sep,
        )
