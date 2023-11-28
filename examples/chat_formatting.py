
import re, regex
from io import StringIO

from pygments import highlight
from pygments.formatter import Formatter
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.style import Style
# from pygments.styles.default import DefaultStyle
from pygments.token import Token
from pygments.util import ClassNotFound

import shutil

# List of languages to detect after ``` delimiter

languages = \
{
    "python": "python",
    "c#": "csharp",
    "csharp": "csharp",
    "c": "c",
    "c++": "c",
    "cpp": "c",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "rust": "rust",
    "ruby": "ruby",
    "go": "go",
    "php": "php",
    "yaml": "yaml",
    "json": "json"
}


# Code block formatter for black background

class BlackBackgroundTerminalFormatter(TerminalFormatter):

    code_pad: int = 2
    block_pad_left: int = 1


    def __init__(self):
        super().__init__(style = "monokai")


    def begin(self):

        self.code_pad = 2
        self.block_pad_left = 1


    def format(self, tokensource, outfile):
        # Create a buffer to capture the parent class's output
        buffer = StringIO()
        # Call the parent class's format method
        super().format(tokensource, buffer)
        # Get the content from the buffer
        content = buffer.getvalue()

        # Padding of code
        lines = content.split('\n')
        padded_lines = [f"{lines[0]}{' ' * self.code_pad * 2}"] + [f"{' ' * self.code_pad}{line}{' ' * self.code_pad}" for line in
                                                                   lines[1:-1]] + [lines[-1]]
        content = '\n'.join(padded_lines)

        # Modify the ANSI codes to include a black background
        modified_content = self.add_black_background(content)

        # Offset codeblock
        modified_content = '\n'.join([modified_content.split('\n')[0]] + [f"{' ' * self.block_pad_left}{line}" for line in
                                                                          modified_content.split('\n')[1:]])

        # Relay the modified content to the outfile
        outfile.write(modified_content)


    def add_black_background(self, content):
        # Split the content into lines
        lines = content.split('\n')

        # Process each line to ensure it has a black background
        processed_lines = []
        for line in lines:
            # Split the line into tokens based on ANSI escape sequences
            tokens = re.split(r'(\033\[[^m]*m)', line)
            # Process each token to ensure it has a black background
            processed_tokens = []
            for token in tokens:
                # If the token is an ANSI escape sequence
                if re.match(r'\033\[[^m]*m', token):
                    # Append the black background code to the existing ANSI code
                    processed_tokens.append(f'{token}\033[40m')
                else:
                    # If the token is not an ANSI escape sequence, add the black background code to it
                    processed_tokens.append(f'\033[40m{token}\033[0m')  # Reset code added here

            # Join the processed tokens back into a single line
            processed_line = ''.join(processed_tokens)
            # Add the ANSI reset code to the end of the line
            processed_line += '\033[0m'
            processed_lines.append(processed_line)

        # Join the processed lines back into a single string
        modified_content = '\n'.join(processed_lines)

        return modified_content


class CodeBlockFormatter:

    code_block_text: str
    lines_printed: int
    #last_lexer: Lexer

    formatter = BlackBackgroundTerminalFormatter()
    held_chunk: str

    lines: list
    formatted_lines: list
    max_line_length: int

    def __init__(self):

        delimiter_exp = r"^```[^\s]*"
        self.delimiter_pattern = regex.compile(delimiter_exp)

        self.held_chunk = ""
        self.formatted_lines = []
        self.lines = []
        self.last_lexer = None
        self.next_explicit_language = None
        self.explicit_language = None
        self.max_line_length = 0


    # Start of format block

    def begin(self):
        global languages

        self.code_block_text = ""
        self.formatted_lines = []
        self.lines = []
        self.lines_printed = 0
        self.max_line_length = 0

        self.explicit_language = self.next_explicit_language
        if self.explicit_language is not None:
            self.last_lexer = get_lexer_by_name(self.explicit_language)
            # print("[" + self.explicit_language + "]", end = "")
        else:
            self.last_lexer = get_lexer_by_name("text")

        self.formatter.begin()


    # Print a code block, updating the CLI in real-time

    def print_code_block(self, chunk):
        terminal_width, terminal_height = shutil.get_terminal_size()

        # Start with a blank line if the lines list is empty
        if len(self.lines) == 0: 
            self.lines = [""]

        # Check if the chunk will exceed the terminal width on the current line
        current_line_length = len(self.lines[-1]) + len(chunk) + 2 * self.formatter.code_pad + self.formatter.block_pad_left
        if current_line_length > terminal_width:
            self.lines.append("")

        # Replace tab characters with spaces
        chunk = chunk.replace("\t", "    ")

        # Update the code block text and the current line
        self.code_block_text += chunk
        self.lines[-1] += chunk

        # Keep track of the longest line
        self.max_line_length = max(self.max_line_length, len(self.lines[-1]))

        # Split and format the line if it contains a newline
        if "\n" in self.lines[-1]:
            split = self.lines[-1].split("\n")
            self.lines[-1] = split[0]

            # Try guessing the lexer for syntax highlighting
            try:
                if self.explicit_language is None and '\n' in chunk: 
                    lexer = guess_lexer(self.code_block_text)
                    self.last_lexer = lexer
                else:
                    lexer = self.last_lexer
            except ClassNotFound:
                lexer = get_lexer_by_name("text")
                self.last_lexer = lexer

            # Determine which lines to display based on terminal height
            if len(self.lines) > terminal_height - 1:
                start_line = len(self.lines) - (terminal_height - 1)
                display_lines = self.lines[start_line:]
            else:
                start_line = 0
                display_lines = self.lines

            # Format the display_lines
            padded_lines = [line.ljust(self.max_line_length) for line in display_lines]
            padded_text = "\n".join(padded_lines)
            highlighted_text = highlight(padded_text, lexer, self.formatter)
            highlighted_text = highlighted_text.replace('\n', '\033[0m\n')

            # Move cursor to the correct position for updating text
            cursor_up_lines = min(len(display_lines) - 1, terminal_height - 1)
            print(f"\x1b[{cursor_up_lines}A\x1b[0G ", end="")

            # Print the formatted lines
            print(highlighted_text, end="")

            # Prepare for the next line
            self.lines.append(split[1])
            print(self.lines[-1], end="")
        else:
            # Print the chunk if it doesn't contain a newline
            print(chunk, end="")



    def process_delimiter(self, chunk):
        global languages

        self.held_chunk += chunk

        match = self.delimiter_pattern.match(self.held_chunk, partial = True)

        # No match, emit any held text

        if not match:
            chunk = self.held_chunk
            self.held_chunk = ""
            return chunk, False

        # Recognize delimited when at least one character of the held chunk doesn't match

        pos = match.end()

        if pos < len(self.held_chunk):
            match_str = self.held_chunk[3:pos]
            chunk = self.held_chunk[pos:]
            self.held_chunk = ""

            if match_str in languages:
                self.next_explicit_language = languages[match_str]
            else:
                self.next_explicit_language = None

            return chunk, True

        # The entire chunk matches, so it may not be complete yet

        return "", False
