
import re
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

    formatter = BlackBackgroundTerminalFormatter()

    # Start of format block

    def begin(self):

        self.code_block_text = ""
        self.lines_printed = 0

        self.formatter.begin()


    # Print a code block, updating the CLI in real-time

    def print_code_block(self, chunk):

        # Clear previously printed lines
        for _ in range(self.lines_printed):  # -1 not needed?
            # Move cursor up one line
            print('\x1b[1A', end='')
            # Clear line
            print('\x1b[2K', end='')

        terminal_width = shutil.get_terminal_size().columns

        # Check if the chunk will exceed the terminal width on the current line
        current_line_length = len(self.code_block_text.split('\n')[-1]) + len(chunk) + 2 * 3 + 3  # Including padding and offset
        if current_line_length > terminal_width:
            self.code_block_text += '\n'

        # Update the code block text
        self.code_block_text += chunk

        # Remove language after codeblock start
        code_block_text = '\n'.join([''] + self.code_block_text.split('\n')[1:])

        specified_lang = self.code_block_text.split('\n', 1)[0]  # Get 1st line (directly after delimiter, can be language)

        # Split updated text into lines and find the longest line
        lines = code_block_text.split('\n')
        max_length = max(len(line) for line in lines)

        # Pad all lines to match the length of the longest line
        padded_lines = [line.ljust(max_length) for line in lines]

        # Join padded lines into a single string
        padded_text = '\n'.join(padded_lines)

        # Try guessing the lexer for syntax highlighting, if we haven't guessed already
        try:
            lexer = guess_lexer(padded_text) if specified_lang is None else get_lexer_by_name(specified_lang)
        except ClassNotFound:
            lexer = get_lexer_by_name("text")  # Fallback to plain text if language isn't supported by pygments

        # Highlight
        highlighted_text = highlight(padded_text, lexer, self.formatter)
        highlighted_text = highlighted_text.replace('\n', '\033[0m\n')

        # Print the updated padded and highlighted text
        print(highlighted_text, end='')

        # Update the lines_printed counter
        self.lines_printed = len(lines)

    def write(self):
        f = open("demofile3.txt", "w")
        f.write(self.code_block_text)
        f.close()
