# SBGTool v2.0
Developed by Zeynab (Artemis) Mohseni, Spring 2022

# What is Similarity-Based Grouping Tool (SBGTool) about?

*SBGTool* is a web-based Visual Learning Analytics (VLA) tool that assists teachers in categorizing students into different groups based on their similar learning outcomes.  Teachers could use SBGTool to:

* Identify the week that has the highest number of interactions for examining the students' engagement
* Find the number of students, correct and incorrect answers in a class over a week or an academic year
* Find out which subjects are the most challenging and which are the simplest in a class over a week or an academic year
* Identify the number of correct and incorrect answers for the most difficult and easiest subject
* Recognize the students with the most answers in different performance levels
* Find the date, hour, and subject with the most interactions from students
* Compare the outcomes of individual students
* Determine the maximum and minimum answer times for each subject and each student throughout the period of a week and an academic year
* Find the student with the most engagement over the period of a week and an academic year using the tool

                                
# What does SBGTool show?

SBGTool is split into three sections: __key metrics__, __overview__, and __detail__. The *key metrics* section gives broad information about the dataset to the teachers. By using *overview* that is a timeline-based section, teachers may get a comprehensive summary of the students' engagements and the number of correct/incorrect answers in different weeks of an academic year. *Detail* section contains a table, two bar charts, and three tabs with different visualizations. Teachers can use the table to filter and sort the features and extract detailed information about one student, a certain week, a subject, a user answer among the four answer choices, a correct answer among the four answer choices, result and the maximum and minimum answer durations. 
The proposed bar charts in the left-hand side of the tool can be used by teachers to find the percentages of correct and incorrect answers, and the most difficult and easitest subjects. Furthermore, the visualizations presented in the three tabs allow teachers to group students based on their similar learning outcomes, compare the outcomes of two individual students, and find students with similar learning activities. 

# Publication about SBGTool v2.0:
https://www.mdpi.com/2306-5729/7/7/98

# Requirements
Python 3.9.7

# How to run this tool?
Create a folder called SBGTool in the root folder of your computer by opening the terminal or command prompt. All the files in this repository should be downloaded and placed in the SBGTool folder.

```
mkdir SBGTool
cd SBGTool
```

Install all required packages by running:

```
pip install -r requirements.txt
```

Run this app locally with:

```
python app.py
```


# Screenshot

![GitHub Logo](/SBGTool.png)
