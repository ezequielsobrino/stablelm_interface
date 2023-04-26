from setuptools import setup, find_packages

setup(
    name="stablelm_interface",
    version="0.1",
    description="provides a simple interface for working with the StableLM language model from StabilityAI. It includes functions for initializing the model, generating text, and customizing the text generation process.",
    author="Ezequiel Sobrino",
    author_email="ezequiel.sobrino@gmail.com",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch"
    ],
)