# Requirements
pip install tensorflow

pip install nltk

pip install numpy

New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Data model
{"tag": "tag",
              "patterns": ["question1", "question2", "question3"],
              "responses": ["réponse1", "réponse2", "réponse3"],
             }

# Apprentissage