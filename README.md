# Codes for Conditional Independence Test by Conditional Flow Models

## Explanations for the files

-   Files with prefix *functions*
    -   functions_gcit.py
        -   Code adapted from: https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/gcit
        -   We revised the code to make the splitting proportion (training/test ratio) as an input to maintain size under $H_0$. For each set of simulation, the ratio is then fixed.
    -   functions_flow.py
        -   The main code to implement our test approach.
    -   functions_generate_data.py