## links

[Yolov8](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb#scrollTo=bpF9-vS_DAaf)

# Jupyter Book template for AIFactory Team KAR1

가짜 연구소에서 Jupyter Book page에 대한 template를 사용했습니다.

저희가 공부도 병행해서 공모전을 준비할 때 책 형태로 만들어서 공부한 것을 정리하고자 사용했습니다. 

각자 branch를 만들고 clone후에 내용을 수정하고 commit 후 full request합니다. 

브랜치 수정내용을 회의때 확인 후에 main에 최종 반영합니다. 

1. make a new repo in PseudoLab Github with the study group name as repo name

   ```
   Pseudo-Lab/2021-Kaggle-Study
   ```

2. clone the repo on your local computer

   ```
   git clone https://github.com/Pseudo-Lab/2021-Kaggle-Study.git
   ```

3. clone this repo

   ```
   git clone https://github.com/Pseudo-Lab/Jupyter-Book-Template
   ```

4. move `book` folder to the `2021-Kaggle-Study` folder which has been created at step 2. 

5. change the contents in `book/docs` folder with the contents from your studies

6. configure `_toc.yml` file

7. build the book using Jupyter Book command

   ```
   jupyter-book build 2021-Kaggle-Study/book
   ```

8. sync your local and remote repositories

   ```
   cd 2021-Kaggle-Study
   git add .
   git commit -m "adding my first book!"
   git push
   ```

9. Publish your Jupyter Book with Github Pages

   ```
   ghp-import -n -p -f book/_build/html -m "initial publishing"
   ```
