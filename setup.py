from setuptools import setup, find_packages

setup(
    name="pact_custom_transformers",  # You can choose any name
    version="0.1",
    packages=find_packages(),       # This will find 'transformers' and any other folder with __init__.py
    include_package_data=True,
    install_requires=[],
    description="Custom version of transformers.",
    author="Dhouib Mohamed",
)
