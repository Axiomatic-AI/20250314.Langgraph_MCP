#!/usr/bin/env python
"""Test script for verifying physics and mathematics derivations.

This script provides tests for various physics and mathematics equations and derivations
using our MathVerificationMCP class that implements improved mathematical verification.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

# Add parent directory to path so we can import modules from parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our MathVerificationMCP class
from advanced_research_modules import MathVerificationMCP, MathematicalVerification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_equation(equation: str, description: str) -> Dict[str, Any]:
    """Test a single equation and print results.
    
    Args:
        equation: The equation to verify
        description: Description of the equation
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"\nTesting: {description}")
    logger.info(f"Equation: {equation}")
    
    # Use our actual implementation
    verifier = MathVerificationMCP()
    result = verifier.verify_equation(equation)
    
    if result["valid"]:
        logger.info("Verification result: Valid")
    else:
        logger.info("Verification result: Invalid")
        logger.info(f"Message: {result['message']}")
    
    return {
        "description": description,
        "equation": equation,
        "result": result
    }

def test_consistency(step1: str, step2: str, description: str) -> Dict[str, Any]:
    """Test the consistency between two derivation steps.
    
    Args:
        step1: First derivation step
        step2: Second derivation step
        description: Description of the test
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"\nTesting Consistency: {description}")
    logger.info(f"Step 1: {step1}")
    logger.info(f"Step 2: {step2}")
    
    # Use our actual implementation
    verifier = MathVerificationMCP()
    result = verifier.check_step_consistency(step1, step2)
    
    if result.get("consistent", False):
        logger.info("Consistency check result: Steps are consistent")
        logger.info(f"Method used: {result.get('method_used', 'unknown')}")
    else:
        logger.info("Consistency check result: Steps are inconsistent")
        logger.info(f"Message: {result.get('message', 'No message provided')}")
    
    return {
        "description": description,
        "step1": step1,
        "step2": step2,
        "result": result
    }

def test_derivation(steps: List[str], description: str) -> Dict[str, Any]:
    """Test a multi-step derivation and print results.
    
    Args:
        steps: List of derivation steps
        description: Description of the derivation
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"\nTesting Derivation: {description}")
    logger.info(f"Steps: {', '.join(steps)}")
    
    # Use our actual implementation
    verifier = MathVerificationMCP()
    result = verifier.verify_derivation(steps)
    
    if result.get("valid", False):
        logger.info("Derivation verification result: Valid")
    else:
        logger.info("Derivation verification result: Invalid")
        if "error" in result:
            logger.info(f"Error: {result['error']}")
    
    return {
        "description": description,
        "steps": steps,
        "result": result
    }

def save_results(results: List[Dict[str, Any]], filename: str) -> None:
    """Save verification results to a JSON file.
    
    Args:
        results: List of verification results
        filename: Name of the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as file:
        json.dump(results, file, indent=2)
    
    logger.info(f"\nResults saved to: {filename}")

def main() -> None:
    """Run verification tests on physics and mathematics derivations."""
    logger.info("Starting improved mathematics verification tests")
    
    results = []
    
    # Test basic equation validation
    equations_to_test = [
        ("x = y + z", "Basic linear equation"),
        ("E = m*c^2", "Einstein's mass-energy equivalence"),
        ("F = G * m1 * m2 / r^2", "Newton's law of gravitation"),
        ("invalid equation", "Invalid equation without equals sign"),
        ("x + y = ", "Invalid equation with empty right side"),
        ("x^2 + y^2 = r^2", "Circle equation"),
        ("a + b = c", "Simple addition equation")
    ]
    
    for equation, description in equations_to_test:
        results.append(test_equation(equation, description))
    
    # Test consistency checks
    consistency_tests = [
        # Consistent transformations
        ("x = y + z", "x - y = z", "Consistent algebraic rearrangement"),
        ("a + b = c", "c - a = b", "Consistent algebraic rearrangement"),
        ("x^2 + y^2 = r^2", "x^2 = r^2 - y^2", "Consistent algebraic rearrangement"),
        ("v = u + a*t", "v - u = a*t", "Consistent kinematics equation"),
        
        # Inconsistent transformations (our focus cases)
        ("x = y + z", "x - z = y + 1", "Inconsistent - adding constant"),
        ("a + b = c", "a = c - b + 2", "Inconsistent - adding constant"),
        ("E = m*c^2", "E/m = c^2 + 1", "Inconsistent - adding term"),
        
        # Edge cases
        ("x = 2*y", "x/2 = y", "Consistent division"),
        ("a*b = c", "a = c/b", "Consistent division"),
        ("x + y = 10", "2*x + 2*y = 20", "Consistent multiplication")
    ]
    
    for step1, step2, description in consistency_tests:
        results.append(test_consistency(step1, step2, description))
    
    # Test multi-step derivations
    # 1. Consistent derivation
    steps_velocity = [
        "a = dv/dt",
        "dv = a * dt",
        "∫dv = ∫a * dt",
        "v - v_0 = a * t",
        "v = v_0 + a * t"
    ]
    results.append(test_derivation(steps_velocity, "Derivation of Velocity Equation"))
    
    # 2. Inconsistent derivation (contains our problem case)
    steps_inconsistent = [
        "x = y + z",
        "x - z = y + 1",  # This step is inconsistent with the previous one
        "x - y = z + 1"   # This also propagates the error
    ]
    results.append(test_derivation(steps_inconsistent, "Inconsistent Derivation Example"))
    
    # Save results to JSON file
    save_results(results, os.path.join(os.path.dirname(__file__), "verification_results", "math_verification_test_results.json"))
    
    logger.info("Testing complete")

if __name__ == "__main__":
    main()
