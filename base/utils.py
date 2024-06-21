#!/usr/bin/env python3
#
import sys, os, pdb
import json, math, re, csv
import random
import signal
from collections import defaultdict

def topological_sort(graph, vertices):
    visited = set()
    stack = []

    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(vertex)

    for vertex in vertices:
        if vertex not in visited:
            dfs(vertex)
    return stack[::-1]


def generate_sample_values(variable_dict):
    # Create a copy of the variable dictionary to store the generated values
    generated_values = variable_dict.copy()

    # Create a dependency graph using a defaultdict
    graph = defaultdict(list)
    vertices = []
    for variable, value_range in variable_dict.items():
        vertices.append(variable)
        for v in variable_dict:
            if v in value_range and v != variable:
                graph[v].append(variable)

    # Perform topological sorting to get the evaluation order
    evaluation_order = topological_sort(graph, vertices)

    # Evaluate the variables in the obtained order
    for variable in evaluation_order:
        value_range = variable_dict[variable]
        locals()[variable] = eval(value_range)
        generated_values[variable] = locals()[variable]
        if isinstance(generated_values[variable], float):
            generated_values[variable] = round(generated_values[variable], 2)
    return generated_values


def run_func(variables={}, function_code="", timeout_duration = 3):
    if not function_code:
        print("Empty function code ...")
        return "Empty function code ..."
    exec(function_code)
    function = locals()['solution']

    # Check if variables match the input parameters of the function
    input_parameters = function.__code__.co_varnames[:function.__code__.co_argcount]  # Get the parameter names of the function
    if not set(input_parameters) <= set(variables.keys()):
        # print("Variables do not match the input parameters of the function.")
        print("Function input: ", input_parameters)
        print("Variables from range: ", variables.keys())
        # pdb.set_trace()
        return "Variables do not match the input parameters of the function."

    variables = {key_:value_ for key_, value_ in variables.items() if key_ in input_parameters}
    # Define the signal handler
    def timeout_handler(signum, frame):
        raise TimeoutError
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the alarm
    signal.alarm(timeout_duration)
    try:
        result = function(**variables)
        return result
    except TimeoutError:
        print(f"Function execution exceeded {timeout_duration} seconds and was terminated.")
        return f"Function execution exceeded {timeout_duration} seconds and was terminated."
    except Exception as err:
        print("Error:", err)
        # pdb.set_trace()
        return str(err)
    finally:
        # Cancel the alarm
        signal.alarm(0)


def is_number(string):
    if type(string) != str:
        string = str(string)
    pattern = re.compile(r'^-?\d+(\.\d+)?$|^-?\d+/\d+$')
    return bool(pattern.match(string))
            
            
def extract_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS