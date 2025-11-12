#!/usr/bin/env python3
"""
Simple TeX verifier: checks for balanced braces, begin/end environment matching,
\input/\include file existence, \includegraphics file existence, duplicate labels,
non-ASCII characters, math mode balance, reference validation, citation checks,
command argument issues, code environments, figure/table references, bibliography
entries, hyperlinks, and custom command definitions. Produces a JSON report and prints a summary.

This script is read-only and will not modify any .tex files.
"""
import re
import sys
import json
from pathlib import Path


def read_tex(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def strip_comments(text: str) -> str:
    # Remove LaTeX comments for analysis (everything after % on a line)
    return re.sub(r'%.*', '', text, flags=re.MULTILINE)


def find_unbalanced_braces(text: str):
    text = strip_comments(text)
    stack = []
    issues = []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if stack:
                stack.pop()
            else:
                issues.append({'pos': i, 'type': 'extra_closing_brace', 'severity': 'error'})
    for pos in stack:
        issues.append({'pos': pos, 'type': 'unclosed_open_brace', 'severity': 'error'})
    return issues


ENV_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
ENV_END = re.compile(r'\\end\{([^}]+)\}')
INCLUDE_CMD = re.compile(r'\\(?:input|include)\{([^}]+)\}')
GRAPHICS_CMD = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
LABEL_CMD = re.compile(r'\\label\{([^}]+)\}')
REF_CMD = re.compile(r'\\ref\{([^}]+)\}')
CITE_CMD = re.compile(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}')
BIBLIOGRAPHY_CMD = re.compile(r'\\bibliography\{([^}]+)\}')
SECTION_CMD = re.compile(r'\\(?:section|subsection|subsubsection|chapter|part)\{([^}]*)\}')


def check_environments(text: str):
    text = strip_comments(text)
    begins = ENV_BEGIN.findall(text)
    ends = ENV_END.findall(text)
    stack = []
    issues = []
    # Simple linear scan
    tokens = list(re.finditer(r'\\begin\{[^}]+\}|\\end\{[^}]+\}', text))
    for m in tokens:
        token = m.group(0)
        if token.startswith('\\begin'):
            name = ENV_BEGIN.match(token).group(1)
            stack.append((name, m.start()))
        else:
            name = ENV_END.match(token).group(1)
            if stack and stack[-1][0] == name:
                stack.pop()
            else:
                issues.append({'pos': m.start(), 'type': 'mismatched_end', 'env': name, 'severity': 'error'})
    for name, pos in stack:
        issues.append({'pos': pos, 'type': 'unclosed_begin', 'env': name, 'severity': 'error'})
    return issues


def check_math_balance(text: str):
    text = strip_comments(text)
    issues = []
    inline_stack = []
    display_stack = []
    for i, ch in enumerate(text):
        if ch == '$':
            if i + 1 < len(text) and text[i + 1] == '$':
                # Display math $$
                if display_stack:
                    display_stack.pop()
                else:
                    display_stack.append(i)
                i += 1  # Skip next $
            else:
                # Inline math $
                if inline_stack:
                    inline_stack.pop()
                else:
                    inline_stack.append(i)
    for pos in inline_stack:
        issues.append({'pos': pos, 'type': 'unclosed_inline_math', 'severity': 'error'})
    for pos in display_stack:
        issues.append({'pos': pos, 'type': 'unclosed_display_math', 'severity': 'error'})
    return issues


def check_references(text: str):
    text = strip_comments(text)
    labels = set(LABEL_CMD.findall(text))
    refs = REF_CMD.findall(text)
    issues = []
    for ref in refs:
        if ref not in labels:
            # Find position of this ref
            m = REF_CMD.search(text)
            if m:
                issues.append({'pos': m.start(), 'type': 'undefined_reference', 'ref': ref, 'severity': 'error'})
                text = text[m.end():]  # Continue searching
    return issues


def check_citations(path: Path, text: str):
    text = strip_comments(text)
    cites = CITE_CMD.findall(text)
    bib_cmd = BIBLIOGRAPHY_CMD.search(text)
    issues = []
    if cites and not bib_cmd:
        # Check for .bib files in directory
        bib_files = list(path.parent.glob('*.bib'))
        if not bib_files:
            issues.append({'type': 'missing_bibliography', 'cites': len(cites), 'severity': 'warning'})
    # Note: Full bib entry validation would require parsing .bib, which is complex; this is a basic check
    return issues


def check_command_args(text: str):
    text = strip_comments(text)
    issues = []
    for m in SECTION_CMD.finditer(text):
        arg = m.group(1).strip()
        if not arg:
            issues.append({'pos': m.start(), 'type': 'empty_section_arg', 'severity': 'warning'})
    return issues


def check_includes(path: Path, text: str):
    text = strip_comments(text)
    missing = []
    for m in INCLUDE_CMD.finditer(text):
        candidate = (path.parent / m)
        if not candidate.exists() and not (candidate.with_suffix('.tex')).exists():
            missing.append({'cmd': 'include/input', 'target': m, 'severity': 'warning'})
    for m in GRAPHICS_CMD.findall(text):
        candidate = (path.parent / m)
        # allow common image extensions
        if not any((candidate.with_suffix(ext)).exists() for ext in ['.png', '.pdf', '.jpg', '.jpeg', '.eps']):
            if not candidate.exists():
                missing.append({'cmd': 'includegraphics', 'target': m, 'severity': 'warning'})
    return missing


def check_labels(text: str):
    text = strip_comments(text)
    labels = LABEL_CMD.findall(text)
    dupes = []
    seen = {}
    for i, lab in enumerate(labels):
        if lab in seen:
            dupes.append({'label': lab, 'first_index': seen[lab], 'second_index': i, 'severity': 'warning'})
        else:
            seen[lab] = i
    return dupes


def check_non_ascii(text: str):
    text = strip_comments(text)
    bad = []
    for i, ch in enumerate(text):
        if ord(ch) > 127:
            bad.append({'pos': i, 'char': ch, 'code': ord(ch), 'severity': 'warning'})
            if len(bad) > 20:
                break
    return bad


# New checks for comprehensive verification
def check_code_environments(text: str):
    text = strip_comments(text)
    issues = []
    # Check for unclosed verbatim, lstlisting, etc.
    code_envs = ['verbatim', 'lstlisting', 'minted', 'python', 'code']
    for env in code_envs:
        begins = list(re.finditer(r'\\begin\{' + re.escape(env) + r'\}', text))
        ends = list(re.finditer(r'\\end\{' + re.escape(env) + r'\}', text))
        if len(begins) != len(ends):
            issues.append({'type': 'unbalanced_code_env', 'env': env, 'begins': len(begins), 'ends': len(ends), 'severity': 'error'})
    return issues


def check_figure_table_refs(text: str):
    text = strip_comments(text)
    issues = []
    # Find figure/table environments and check for labels
    fig_matches = list(re.finditer(r'\\begin\{figure\}', text))
    tab_matches = list(re.finditer(r'\\begin\{table\}', text))
    for matches, env_type in [(fig_matches, 'figure'), (tab_matches, 'table')]:
        for m in matches:
            # Check if there's a label in the environment
            end_pos = text.find(r'\end{' + env_type + '}', m.end())
            if end_pos != -1:
                env_content = text[m.end():end_pos]
                if not re.search(r'\\label\{[^}]+\}', env_content):
                    issues.append({'pos': m.start(), 'type': 'missing_label', 'env': env_type, 'severity': 'warning'})
    return issues


def check_bibliography_entries(path: Path, text: str):
    text = strip_comments(text)
    issues = []
    bib_cmd = BIBLIOGRAPHY_CMD.search(text)
    if bib_cmd:
        bib_file = bib_cmd.group(1)
        bib_path = path.parent / (bib_file + '.bib')
        if bib_path.exists():
            bib_content = bib_path.read_text(encoding='utf-8', errors='replace')
            cites = set(CITE_CMD.findall(text))
            bib_entries = set(re.findall(r'@\w+\{([^,]+),', bib_content))
            missing = cites - bib_entries
            for miss in missing:
                issues.append({'type': 'missing_bib_entry', 'cite': miss, 'severity': 'error'})
        else:
            issues.append({'type': 'missing_bib_file', 'file': bib_file + '.bib', 'severity': 'error'})
    return issues


def check_hyperlinks(text: str):
    text = strip_comments(text)
    issues = []
    # Check for broken \url or \href
    url_matches = re.findall(r'\\url\{([^}]+)\}', text)
    href_matches = re.findall(r'\\href\{([^}]+)\}\{([^}]+)\}', text)
    for url in url_matches + [h[0] for h in href_matches]:
        if not url.startswith(('http://', 'https://', 'ftp://')):
            issues.append({'type': 'invalid_url', 'url': url, 'severity': 'warning'})
    return issues


def check_custom_commands(text: str):
    text = strip_comments(text)
    issues = []
    # Check for undefined custom commands (basic heuristic)
    newcommand_matches = re.findall(r'\\newcommand\{\\([^}]+)\}', text)
    defined_cmds = set(newcommand_matches)
    all_cmds = set(re.findall(r'\\([a-zA-Z]+)', text))
    undefined = all_cmds - defined_cmds - {'begin', 'end', 'section', 'subsection', 'label', 'ref', 'cite', 'includegraphics', 'bibliography', 'url', 'href'}  # Common built-ins
    for cmd in undefined:
        if len(cmd) > 1:  # Avoid single letters
            issues.append({'type': 'undefined_command', 'cmd': cmd, 'severity': 'warning'})
    return issues


def run_checks(tex_path: Path):
    text = read_tex(tex_path)
    report = {}
    report['path'] = str(tex_path)
    report['unbalanced_braces'] = find_unbalanced_braces(text)
    report['env_issues'] = check_environments(text)
    report['math_balance'] = check_math_balance(text)
    report['ref_issues'] = check_references(text)
    report['cite_issues'] = check_citations(tex_path, text)
    report['cmd_arg_issues'] = check_command_args(text)
    report['missing_includes'] = check_includes(tex_path, text)
    report['duplicate_labels'] = check_labels(text)
    report['non_ascii'] = check_non_ascii(text)
    # New comprehensive checks
    report['code_env_issues'] = check_code_environments(text)
    report['fig_tab_issues'] = check_figure_table_refs(text)
    report['bib_issues'] = check_bibliography_entries(tex_path, text)
    report['hyperlink_issues'] = check_hyperlinks(text)
    report['custom_cmd_issues'] = check_custom_commands(text)
    # Calculate summary with severity
    errors = sum(len(report[k]) for k in ['unbalanced_braces', 'env_issues', 'math_balance', 'ref_issues', 'code_env_issues', 'bib_issues'])
    warnings = sum(len(report[k]) for k in ['cite_issues', 'cmd_arg_issues', 'missing_includes', 'duplicate_labels', 'non_ascii', 'fig_tab_issues', 'hyperlink_issues', 'custom_cmd_issues'])
    report['summary'] = {'errors': errors, 'warnings': warnings, 'total_issues': errors + warnings}
    return report


def main(argv):
    if len(argv) < 2:
        print('Usage: tex_verify.py file.tex')
        return 2
    p = Path(argv[1])
    if not p.exists():
        print('File not found:', p)
        return 2
    report = run_checks(p)
    out_path = p.parent / (p.stem + '.tex_verify.json')
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Verification saved to', out_path)
    # Print compact summary
    print('Summary: errors={errors}, warnings={warnings}, total={total_issues}'.format(**report['summary']))
    if report['unbalanced_braces']:
        print('Unbalanced braces:', len(report['unbalanced_braces']))
    if report['env_issues']:
        print('Environment issues:', len(report['env_issues']))
    if report['math_balance']:
        print('Math balance issues:', len(report['math_balance']))
    if report['ref_issues']:
        print('Reference issues:', len(report['ref_issues']))
    if report['cite_issues']:
        print('Citation issues:', len(report['cite_issues']))
    if report['cmd_arg_issues']:
        print('Command argument issues:', len(report['cmd_arg_issues']))
    if report['missing_includes']:
        print('Missing includes/graphics:', len(report['missing_includes']))
    if report['duplicate_labels']:
        print('Duplicate labels:', len(report['duplicate_labels']))
    if report['non_ascii']:
        print('Non-ASCII characters:', len(report['non_ascii']))
    # New summary prints
    if report['code_env_issues']:
        print('Code environment issues:', len(report['code_env_issues']))
    if report['fig_tab_issues']:
        print('Figure/table issues:', len(report['fig_tab_issues']))
    if report['bib_issues']:
        print('Bibliography issues:', len(report['bib_issues']))
    if report['hyperlink_issues']:
        print('Hyperlink issues:', len(report['hyperlink_issues']))
    if report['custom_cmd_issues']:
        print('Custom command issues:', len(report['custom_cmd_issues']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
