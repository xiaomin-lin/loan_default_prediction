from setuptools import find_packages, setup

setup(
    name="loan_default_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Read dependencies from requirements.txt
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Xiaomin Lin",
    description="Loan default prediction project",
    python_requires=">=3.7",
)
