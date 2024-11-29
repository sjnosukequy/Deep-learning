## 1. REQUIREMENT
1. **Python 3.10+**
1. **Poetry**

## 2. Installation
### 2.1 Install pipx
[pipx Document](https://pipx.pypa.io/stable/installation/)

- **On Windows**

Install via pip (requires pip 19.0 or later)

``` powershell 
py -m pip install --user pipx
```

move to <USER folder>\AppData\Roaming\Python\Python3x\Scripts

``` powershell
.\pipx.exe ensurepath
```
This will add both the above-mentioned path and the %USERPROFILE%\.local\bin folder to your search path. Restart your terminal session and verify that pipx runs correctly.

### 2.2 Installing Poetry
[Poetry Document](https://python-poetry.org/docs/#installing-with-pipx)
``` powershell
pipx install poetry
```

### 2.3 Config Poetry to create venv at root
run this command to make poetry to create venv at root
``` powershell
poetry config virtualenvs.in-project true
```

### 2.4 Installing dependencies
move to root project
``` powershell
poetry install
```

### 2.5 Access the venv
``` powershell
poetry shell
```

### 2.6 Run

``` powershell
py Model_1.py
```


### 3 Moddel architecture
Model 1
```mermaid
---
config:
  look: classic
  theme: neo
  layout: dagre
---
flowchart TD
    A["12 * 224 * 224"] --> B["Node 1"]
    B --> C["Conv7x7 P=3 C=32 S=1"]
    C --> n1["MaxPool 5x5 P=0 S=3"]
    n1 --> n2["Conv3x3 P=5 C=32 S=3"] & n14["SUM"]
    n2 --> n3["Conv3x3 P=1 C=16 S=1"] & n4["Conv3x3 P=1 C=16 S=1"] & n5["Conv3x3 P=1 C=16 S=1"] & n6["Conv3x3 P=1 C=16 S=1"]
    n3 --> n7["Conv3x3 P=1 C=16 S=1"]
    n4 --> n8["Conv3x3 P=1 C=16 S=1"]
    n5 --> n9["Conv3x3 P=1 C=16 S=1"]
    n6 --> n10["Conv3x3 P=1 C=16 S=1"]
    n7 --> n11["CAT"]
    n8 --> n11
    n9 --> n12["CAT"]
    n10 --> n12
    n11 --> n13["SUM"]
    n12 --> n13
    n13 --> n14
    n14 --> n15["AvgPool 2x2 P=0 S=2"]
    n15 --> n16["Conv3x3 P=1 C=780 S=0"]
    n16 --> n17["AvgPool 2x2 P=0 S=4"]
    n17 --> n18["Fully Connected 780"]
    n18 --> n19["Fully Connected 3"]
    style A fill:#FFD600
```

Model 2
```mermaid
---
config:
  look: classic
  theme: neo
  layout: dagre
---
flowchart TD
    A["12 * 224 * 224"] --> B["Node 1"]
    B --> C["Conv7x7 P=3 C=32 S=1"]
    C --> n1["AvgPool 5x5 P=0 S=3"]
    n1 --> n2["Conv5x5 P=6 C=32 S=3"] & n14["SUM"]
    n2 --> n3["Conv5x5 P=2 C=16 S=1"] & n4["Conv5x5 P=2 C=16 S=1"] & n5["Conv5x5 P=2 C=16 S=1"] & n6["Conv5x5 P=2 C=16 S=1"]
    n3 --> n7["Conv5x5 P=2 C=16 S=1"]
    n4 --> n8["Conv5x5 P=2 C=16 S=1"]
    n5 --> n9["Conv5x5 P=2 C=16 S=1"]
    n6 --> n10["Conv5x5 P=2 C=16 S=1"]
    n7 --> n11["CAT"]
    n8 --> n11
    n9 --> n12["CAT"]
    n10 --> n12
    n11 --> n13["SUM"]
    n12 --> n13
    n13 --> n14
    n14 --> n15["MaxPool 2x2 P=0 S=2"]
    n15 --> n16["Conv5x5 P=2 C=780 S=0"]
    n16 --> n17["MaxPool 2x2 P=0 S=4"]
    n17 --> n18["Fully Connected 780"]
    n18 --> n19["Fully Connected 3"]
    style A fill:#FFD600
```




