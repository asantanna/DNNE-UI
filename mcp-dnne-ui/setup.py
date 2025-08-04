from setuptools import setup, find_packages

setup(
    name="dnne-ui-mcp",
    version="1.0.0",
    description="DNNE UI MCP Server for Claude Desktop",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mcp>=1.12.0",
        "playwright>=1.54.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dnne-ui-mcp=dnne_ui_mcp_server:main",
        ],
    },
)