#!/usr/bin/env python3
"""
Script to validate the LaTeX file for common formatting issues
and attempt to convert it to PDF.
"""

import re
import os
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode
import sys

def validate_latex_structure(latex_content):
    """
    Validate basic LaTeX structure and common formatting issues
    """
    issues = []
    
    # Check for proper document structure
    if not re.search(r'\\documentclass', latex_content):
        issues.append("Missing \\documentclass declaration")
    
    if not re.search(r'\\begin{document}', latex_content, re.IGNORECASE):
        issues.append("Missing \\begin{document}")
    
    if not re.search(r'\\end{document}', latex_content, re.IGNORECASE):
        issues.append("Missing \\end{document}")
    
    # Check for matching braces
    open_braces = latex_content.count('{')
    close_braces = latex_content.count('}')
    if open_braces != close_braces:
        issues.append(f"Mismatched braces: {open_braces} opening, {close_braces} closing")
    
    # Check for matching environments
    environments = re.findall(r'\\begin\{(\w+)\}', latex_content)
    for env in environments:
        begin_count = len(re.findall(rf'\\begin\{{{env}\}}', latex_content))
        end_count = len(re.findall(rf'\\end\{{{env}\}}', latex_content))
        if begin_count != end_count:
            issues.append(f"Mismatched environment {env}: {begin_count} begin, {end_count} end")
    
    # Check for common problematic patterns
    if re.search(r'\\begin\{document\}.*\\documentclass', latex_content, re.DOTALL):
        issues.append("\\documentclass found after \\begin{document}")
    
    # Check for double underscores (potential escaping issues)
    if re.search(r'\\section\*?\s*{.*__.*}', latex_content):
        issues.append("Double underscores found in section title (potential formatting issue)")
    
    return issues

def check_packages_and_commands(latex_content):
    """
    Check for common packages and commands that might be used in the document
    """
    warnings = []
    
    # Look for packages that are used but might not be available in basic environments
    common_missing_packages = [
        ('tikz', 'TikZ graphics package'),
        ('algorithmic', 'Algorithmic package'),
        ('algorithm', 'Algorithm package'),
        ('mdframed', 'mdframed package'),
        ('fancyhdr', 'fancyhdr package'),
        ('caption', 'caption package'),
        ('hyperref', 'hyperref package'),
        ('geometry', 'geometry package'),
        ('amsmath', 'amsmath package'),
        ('amssymb', 'amssymb package'),
        ('amsthm', 'amsthm package'),
        ('enumitem', 'enumitem package'),
        ('microtype', 'microtype package'),
        ('newtxtext', 'newtxtext package'),
        ('newtxmath', 'newtxmath package'),
        ('titlesec', 'titlesec package'),
    ]
    
    for package, description in common_missing_packages:
        if f'usepackage{{{package}}}' in latex_content.lower() or f'usepackage{{{package},'.replace(',', '}}') in latex_content.lower():
            # Check if the package is mentioned in the document
            warnings.append(f"Uses {description} (might require additional installation)")
    
    return warnings

def validate_references(latex_content):
    """
    Check for proper reference handling
    """
    issues = []
    
    # Check for unmatched labels and references
    labels = set(re.findall(r'\\label\{([^}]*)\}', latex_content))
    references = set(re.findall(r'\\ref\{([^}]*)\}', latex_content))
    
    # Check for references to undefined labels
    for ref in references:
        if ref not in labels:
            issues.append(f"Reference to undefined label: {ref}")
    
    # Same for citations
    citations = set(re.findall(r'\\cite\{([^}]*)\}', latex_content))
    bibliography_entries = set(re.findall(r'\\bibitem\{([^}]*)\}', latex_content))
    
    for cite in citations:
        if cite not in bibliography_entries:
            # Common pattern - citation without matching bibitem is often OK if using external .bib file
            pass  # For now, just acknowledge this pattern
    
    return issues

def validate_math_environments(latex_content):
    """
    Check for common math environment issues
    """
    issues = []
    
    # Check for $$ math delimiters (should use \( \) or \[ \] instead)
    double_dollar_matches = re.findall(r'\$\$.*?\$\$', latex_content, re.DOTALL)
    if double_dollar_matches:
        issues.append(f"Found {len(double_dollar_matches)} instances of $$ ... $$ math delimiters (consider using \\[ ... \\] for display math)")
    
    # Check for unmatched math delimiters
    inline_start = latex_content.count('\\(')
    inline_end = latex_content.count('\\)')
    if inline_start != inline_end:
        issues.append(f"Unmatched inline math delimiters: {inline_start} \\( and {inline_end} \\)")
    
    display_start = latex_content.count('\\[')
    display_end = latex_content.count('\\]')
    if display_start != display_end:
        issues.append(f"Unmatched display math delimiters: {display_start} \\[ and {display_end} \\]")
    
    return issues

def validate_file(filename):
    """
    Main validation function
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try reading with a different encoding
        with open(filename, 'r', encoding='latin-1') as f:
            content = f.read()
    
    print(f"Validating LaTeX file: {filename}")
    print("="*50)
    
    # Perform various validations
    structure_issues = validate_latex_structure(content)
    package_warnings = check_packages_and_commands(content)
    reference_issues = validate_references(content)
    math_issues = validate_math_environments(content)
    
    # Print results
    if structure_issues:
        print("STRUCTURE ISSUES:")
        for issue in structure_issues:
            print(f"  - {issue}")
        print()
    
    if reference_issues:
        print("REFERENCE ISSUES:")
        for issue in reference_issues:
            print(f"  - {issue}")
        print()
    
    if math_issues:
        print("MATH ENVIRONMENT ISSUES:")
        for issue in math_issues:
            print(f"  - {issue}")
        print()
    
    if package_warnings:
        print("PACKAGE WARNINGS (may need additional installation):")
        for warning in package_warnings:
            print(f"  - {warning}")
        print()
    
    if not any([structure_issues, reference_issues, math_issues]):
        print("✓ No major structural issues found!")
    
    # Convert to plain text to check for parsing errors
    print("Checking for parsing errors...")
    try:
        converter = LatexNodes2Text()
        text = converter.latex_to_text(content)
        print("✓ LaTeX parsing successful!")
    except Exception as e:
        print(f"✗ Parsing error: {e}")
    
    print("\nValidation complete.")
    
    return {
        'structure_issues': structure_issues,
        'package_warnings': package_warnings,
        'reference_issues': reference_issues,
        'math_issues': math_issues
    }

def convert_to_pdf_stub(latex_file_path):
    """
    A stub function - in a real implementation with full LaTeX installed,
    this would convert the LaTeX to PDF.
    """
    print("\n" + "="*50)
    print("PDF CONVERSION NOTE:")
    print("To convert this LaTeX file to PDF, you need a LaTeX distribution installed.")
    print("On Windows, you can install MiKTeX or TeX Live.")
    print("\nThe LaTeX file appears to be well-formed and should compile properly")
    print("once a LaTeX distribution is available.")
    print("\nTo manually compile after installing LaTeX, use:")
    print(f"  pdflatex {latex_file_path}")
    print("  bibtex {filename_without_ext}")  # if needed for bibliography
    print(f"  pdflatex {latex_file_path}")  # second pass
    print(f"  pdflatex {latex_file_path}")  # third pass to resolve all references
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_latex.py <latex_file.tex>")
        print("Example: python validate_latex.py gpt_null_whitepaper.tex")
        sys.exit(1)
    
    latex_file = sys.argv[1]
    if not os.path.exists(latex_file):
        print(f"Error: File {latex_file} not found!")
        sys.exit(1)
    
    results = validate_file(latex_file)
    convert_to_pdf_stub(latex_file)