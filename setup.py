from setuptools import setup, find_packages

setup(
    name="Chain",
    version="2.5",
    packages=find_packages(include=["chain", "chain.*"]),
    entry_points={
        "console_scripts": [
            "update_ollama=Chain.scripts.update_ollama_list:main",
            "chat=Chain.chat.chat:main",
        ],
    },
)
