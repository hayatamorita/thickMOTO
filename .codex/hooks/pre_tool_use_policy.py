import json
import re
import shlex
import sys


OPERATORS = {"|", "||", "&&", ";", "&"}


def split_shell(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def command_arguments(tokens: list[str], index: int) -> list[str]:
    arguments = []
    for argument in tokens[index + 1 :]:
        if argument in OPERATORS:
            break
        arguments.append(argument)
    return arguments


def is_rm_rf(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token != "rm":
            continue
        recursive = False
        force = False
        for argument in command_arguments(tokens, index):
            if argument == "--recursive":
                recursive = True
            elif argument == "--force":
                force = True
            elif argument.startswith("-") and not argument.startswith("--"):
                flags = argument[1:]
                recursive |= "r" in flags or "R" in flags
                force |= "f" in flags
        if recursive and force:
            return True
    return False


def is_curl_pipe_shell(tokens: list[str]) -> bool:
    shells = {"sh", "bash", "dash", "zsh", "ksh"}
    for index, token in enumerate(tokens):
        if token != "curl":
            continue
        for pipe_index in range(index + 1, len(tokens) - 1):
            if tokens[pipe_index] != "|":
                continue
            shell = tokens[pipe_index + 1].rsplit("/", 1)[-1]
            if shell in shells:
                return True
    return False


def is_force_push(tokens: list[str]) -> bool:
    for index in range(len(tokens) - 1):
        if tokens[index : index + 2] != ["git", "push"]:
            continue
        for argument in command_arguments(tokens, index + 1):
            if argument in {"-f", "--force"} or argument.startswith("--force="):
                return True
    return False


def is_chmod_777(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token != "chmod":
            continue
        for argument in command_arguments(tokens, index):
            if re.fullmatch(r"[0-7]*777", argument):
                return True
    return False


def blocked_reason(command: str) -> str | None:
    try:
        tokens = split_shell(command)
    except ValueError:
        return "解析できないシェルコマンドを拒否しました。"

    checks = (
        ("sudo" in tokens, "sudo の実行は禁止されています。"),
        (is_rm_rf(tokens), "rm の再帰・強制削除は禁止されています。"),
        (is_curl_pipe_shell(tokens), "curl の出力をシェルへ直接渡す操作は禁止されています。"),
        (is_force_push(tokens), "git push の強制更新は禁止されています。"),
        (is_chmod_777(tokens), "chmod 777 相当の権限設定は禁止されています。"),
    )
    return next((reason for blocked, reason in checks if blocked), None)


def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        return 0

    command = payload.get("tool_input", {}).get("command", "")
    reason = blocked_reason(command)
    if reason is None:
        return 0

    print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
