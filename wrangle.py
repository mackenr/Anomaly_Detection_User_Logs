from __future__ import division

import os
import warnings
from statistics import mode

import matplotlib.pyplot as plt
import pandas as pd
from env import host, password, username
from IPython.display import display
from numpy import NaN

warnings.filterwarnings("ignore")


def get_db_url(database, host=host, user=username, password=password):
    return f"mysql+pymysql://{user}:{password}@{host}/{database}"


def schema_structure():
    schema = "curriculum_logs"

    query_2 = f"""
    SELECT table_name AS "Tables",
    ROUND(((data_length + index_length) / 1024 / 1024), 4) AS "Size (MB)"
    FROM information_schema.TABLES
    WHERE table_schema = "{schema}"
    ORDER BY (data_length + index_length) DESC;
    """

    info2 = pd.read_sql(query_2, get_db_url(schema))

    tablenames = [x[0] for x in [list(i) for i in info2.values]]

    x = [(pd.read_sql(f"describe {x}", get_db_url(schema))) for x in tablenames]

    display(f"In {schema} you have the following table names and their sizes:", info2)
    [display(i) for i in x]


def anonymized_curriculum_access():
    df = pd.read_table(
        "anonymized-curriculum-access.txt",
        sep="\s",
        header=None,
        names=["date", "time", "page", "id", "cohort", "ip"],
    )

    df["datetime"] = df.date + " " + df.time

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    # the / in page seems to be an error so I am droping those rows from the df for now.

    df = df[df.page != "/"]
    return df


def curriculum_logs_new():
    """
    This function reads the curriculum data from the Codeup db into a df.
    """
    schema = "curriculum_logs"
    sql_query = f""" 

    select * from logs as lg
    left join cohorts as cht
    on
    lg.cohort_id = cht.id
    """

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url(schema))

    return df


def get_curriculum_logs():
    """
    This function reads in data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    """
    csvname = "curriculum_logs.csv"
    if os.path.isfile(csvname):

        # If csv file exists read in data from csv file.
        df = pd.read_csv(csvname, index_col=0)

    else:

        # Read fresh data from db into a DataFrame
        df = curriculum_logs_new()

        # Cache data
        df.to_csv(csvname)

    return df


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def lessons(df):
    """
    assigns a lessons col based on the logic below

    """
    lessons = df.page.value_counts()
    # lessons = lessons[
    #     lessons > lessons.quantile(0.6)
    # ]  # lessons will more than likely be viewed in the top 40 percentile
    lessons = lessons.index.to_list()
    suffixes = [
        ".com",
        ".py",
        ".md",
        ".json",
        ".tex",
        ".com",
        ".jpeg",
        ".jpg",
        ".html",
        ".ico",
        ".pn",
        ".png",
        ".svg",
        ".csv",
        ".org",
        ".git",
        ".gitignore",
        ".php",
        ".zip",
    ]  # lessons are likely not files so I exuded files names
    lessons = list(
        set(
            [
                (l.split("/"))[-1]
                for l in lessons
                if type(l) != int and all(l.find(suf) == -1 for suf in suffixes)
            ]
        )
    )

    additional = [
        "0_Classification_Algorithms",
        "1-fundamentals/1-fundamentals-overview",
        "1-fundamentals/1.1-intro-to-data-science",
        "1-fundamentals/1.2-data-science-pipeline",
        "1-fundamentals/1.3-pipeline-demo",
        "1-fundamentals/2.1-intro-to-excel",
        "1-fundamentals/2.2-excel-functions",
        "1-fundamentals/2.3-visualization-with-excel",
        "1-fundamentals/2.4-more-excel-features",
        "1-fundamentals/3-vocabulary",
        "1-fundamentals/pipeline-demo",
        "1-fundamentals/project",
        "1._Fundamentals",
        "10-anomaly-detection/1-overview",
        "10.01_Acquire_WebScraping",
        "10.02.01_ParseText",
        "10.02.02_POSTagging",
        "10.02.03_TFIDF",
        "10.03_Explore",
        "10.04.01_FeatureExtraction_FreqBased",
        "10.04.02_FeatureExtraction_Word2Vec",
        "10.04.03_SentimentAnalysis",
        "10.04.04_TextClassification",
        "10.04.05_TopicModeling",
        "10.10_Exercises",
        "10.2_Regex",
        "10._NLP",
        "11-nlp/1-overview",
        "11.00_Intro",
        "11.01.01_ConnectingToSpark",
        "11.01.02_DataAcquisition",
        "11._DistributedML",
        "115",
        "12.01_SocialNetworkAnalysis",
        "12.02_Recommenders",
        "12.3.6_Page_Styling",
        "13-advanced-topics/1-tidy-data",
        "13.01.01_Understand",
        "13.01.02.01_Prep",
        "13.01.02.02_TalkAndListen",
        "13.01.02.03_Sketch",
        "13.01.02.04_Prototype",
        "13.01.02_Create",
        "13.01.03_Refine",
        "13.01.04_Present",
        "13.02_Tableau",
        "13.03_Bokeh",
        "13.11_Exercises",
        "13.5_Tableau",
        "13._Storytelling",
        "2-stats/1-overview",
        "2-stats/2.1-intro-to-excel",
        "2-stats/2.2-excel-functions",
        "2-stats/2.2-navigating-excel",
        "2-stats/3.1-descriptive-stats",
        "2-storytelling/1-overview",
        "2-storytelling/2.1-understand",
        "2-storytelling/2.2-create",
        "2-storytelling/3-tableau",
        "2-storytelling/bad-charts",
        "2-storytelling/project",
        "2.00.00_Excel_Prob_Stats",
        "2.00.01_Intro_Excel",
        "2.00.02_Navigating_Excel",
        "2.00.04_PrepareData_Excel",
        "2.00.05_Charts_PivotTables_Sparklines",
        "2.01.00_Descriptive_Stats",
        "2.02.00_Inferential_Stats",
        "2.02.01_Probability",
        "2.02.02_Sampling",
        "2.02.03_Power_Analysis",
        "2.02.04_Distribution_and_Test",
        "2.02.05_Compare_Means",
        "2.02.06_Correlation",
        "2.0_Intro_Stats",
        "3.0-mysql-overview",
        "3.1-mysql-introduction",
        "3.10-more-exercises",
        "3.2-databases",
        "3.3-tables",
        "3.4-basic-statements",
        "3.5.0-clauses-overview",
        "3.5.1-where",
        "3.5.2-order-by",
        "3.5.3-limit",
        "3.6-functions",
        "3.7-group-by",
        "3.8.0-relationships-overview",
        "3.8.1-indexes",
        "3.8.2-joins",
        "3.8.3-subqueries",
        "3.9-temporary-tables",
        "4-python/7.1-ds-libraries-overview",
        "4-python/7.2-intro-to-matplotlib",
        "4-python/7.3-intro-to-numpy",
        "4-python/7.4-intro-to-pandas",
        "4-python/intro-to-sklearn",
        "4-python/project",
        "4.0_overview",
        "4.1_introduction",
        "4.2_data_types_and_variables",
        "4.3_control_structures",
        "4.4_functions",
        "4.5_imports",
        "4.6.0_DS_Libraries_Overview",
        "4.6.1_introduction_to_matplotlib",
        "4.6.2_introduction_to_numpy",
        "4.6.3_introduction_to_pandas",
        "4.6.4_introduction_to_seaborn",
        "4_Matplotlib_Styles",
        "4_Objects",
        "4_introduction_to_sklearn",
        "5-regression/1-overview",
        "5-stats/1-overview",
        "5-stats/2-descriptive-stats",
        "5.00_Intro",
        "5.01_Acquire",
        "5.02_Prep",
        "5.03_Explore",
        "5.04.02_OrdinaryLeastSquares",
        "5.04.03_RidgeRegression",
        "5.04.04_LeastAngleRegression",
        "5.04.05_Exercises",
        "5.05_Deliver",
        "5.0_Intro_Regression",
        "5._Regression",
        "6-regression/1-overview",
        "6-regression/3.1-acquire-and-prep",
        "6.00_Intro",
        "6.01.01_AcquireSQL",
        "6.01.02_Acquirecsv",
        "6.01.03_Summarize",
        "6.02.01_Prep",
        "6.02.02_MissingVals",
        "6.03_Explore",
        "6.04.01_Preprocessing",
        "6.04.02_DecisionTree",
        "6.04.03_KNN",
        "6.04.04_LogisticRegression",
        "6.04.05_SVM",
        "6.04.06_RandomForest",
        "6.04.07_Ensemble",
        "6.05_Deliver",
        "6._Classification",
        "7-classification/6.4-knn",
        "7.00_Intro",
        "7.01_Acquire",
        "7.02_Prep",
        "7.03_Explore",
        "7.04.01_Partitioning",
        "7.04.02_Hierarchical",
        "7.0_Intro_Clustering",
        "7._Clustering",
        "8.00_Intro",
        "8.01_Acquire",
        "8.02_Prep",
        "8.03_Explore",
        "8.04.02_ParametricModeling",
        "8.04.03_MLModeling",
        "8.05_Deliver",
        "8.0_Intro_Module",
        "8.1_Overview",
        "8._Time_Series",
        "9.1_About",
        "9.20_Data",
        "9._Anomaly_Detection",
        "About_NLP",
        "Appendix_Tidy_Data",
        "Dataset_Challenge",
        "Excel_Shortcuts",
        "Exercises",
        "Intro_to_Regression",
        "Intro_to_Regression_Module",
        "Module_6_Classification",
        "Pipeline_Demo",
    ]

    len(lessons)
    df.page = df.page.astype(str)
    df["lesson"] = df.page.apply(
        lambda x: True
        if (
            any([((x.split("/"))[-1] == i) for i in lessons])
            or (any([l == x for l in additional]))
        )
        else False
    )
    return df


def final_merged():
    df = anonymized_curriculum_access()
    df = df.reset_index()
    cur_logs = get_curriculum_logs()
    programs = cur_logs[["program_id", "cohort_id", "start_date", "end_date", "name"]]

    programs_df = (
        programs.groupby(["name", "cohort_id", "start_date", "end_date", "program_id"])
        .sum()
        .reset_index()
    )

    programs_df = programs_df.fillna(0)
    df = df.fillna(0)
    a = set(programs_df.cohort_id)
    b = set(df.cohort)
    c = a.intersection(b)
    d = a - c
    programs_df = programs_df[programs_df.cohort_id != 5]
    programs_df = programs_df.rename(columns={"cohort_id": "cohort"})
    a = programs_df.cohort.to_list()
    b = programs_df.name.to_list()
    c = programs_df.program_id.to_list()
    chortnamemap = dict(zip(a, b))
    chortprogmap = dict(zip(a, c))
    df["program_id"] = df["cohort"].map(chortprogmap)
    df["name"] = df["cohort"].map(chortnamemap)
    nomenclaturemap = {
        3: "DS",
        2: "Full_Stack_PHP",
        1: "Full_Stack_PHP",
        4: "Front_End",
    }

    df["program_name"] = df["program_id"].map(nomenclaturemap)
    df = df.merge(programs_df, how="left", on=["name", "program_id", "cohort"])
    df.start_date = pd.to_datetime(df.start_date + " 00:00:00")
    df.end_date = pd.to_datetime(df.end_date + " 00:00:00")
    past_graduate = (df.end_date - df.datetime).dt.days
    df["from_graduation"] = past_graduate
    inside = df.datetime.between(left=df.start_date, right=df.end_date)
    df["cohort_window"] = inside

    df = lessons(df)

    return df


def programs_dict():
    """partitions three programs and returns a dictionary of separate dfs and the original"""
    merged = final_merged()
    merged.fillna(0)

    ds = merged[merged.program_name == "DS"]
    front_end = merged[merged.program_name == "Front_End"]
    full_stack = merged[merged.program_name == "Full_Stack_PHP"]
    programs = {
        "All": merged,
        "DS": ds,
        "Front_End": front_end,
        "Full_Stack_PHP": full_stack,
    }
    print(programs.keys())
    return programs


def df_union(df):
    dfunion = set(df[df.lesson == True].page)
    df_names = list(df.name.unique())
    for name in df_names:
        print()
        dfunion.intersection_update(
            set(df[(df.name == name) & (df.lesson == True)].page)
        )
        if len(dfunion) == 0:
            print(f"breaks on {name}")
            break
        # print(name)
    df_inter = dfunion
    return df_inter


def q1(df):
    q1 = df[df.lesson == True].page.value_counts().nlargest(n=1)
    newdf = (
        pd.DataFrame(q1)
        .reset_index()
        .rename(columns={"page": "count", "index": "lesson"})
    )
    dfname = df.program_name.iloc[0]
    return newdf.style.set_caption(f"Q1 {dfname}")


def question2(df):
    """we group by the lessons then"""
    true_df = df[(df.lesson == True)]
    dfname = true_df.program_name.iloc[0]

    vals = dict(true_df.page.value_counts())
    true_df["page_counts"] = true_df.page.apply(lambda x: vals.get(x))
    true_df = true_df[true_df.page_counts > true_df.page_counts.quantile(0.85)]
    grp1 = true_df.groupby("page")
    zscoreranks = []
    for group in grp1:
        df = group[-1]
        page = f"{group[0]}"
        df = (
            pd.DataFrame(df.groupby("name").name.count())
            .rename(columns={"name": "count"})
            .reset_index()
        )
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        # display(df.style.set_caption(page))
        mean = df["count"].mean()
        max = df["count"].max()
        std = df["count"].std()
        max_z_score = (max - mean) / std

        name = df.name.iat[0]
        zscoreranks.append((page, max_z_score, name, max))

    final_df = (
        pd.DataFrame(zscoreranks, columns=["lesson", "max_z_score", "name", "count"])
        .nlargest(columns="max_z_score", n=15)
        .reset_index(drop=True)
    )
    return final_df.style.set_caption(f"Q2 {dfname}")


def question3(df):
    q3 = df[(df.cohort_window == True) & (df.lesson == True)]
    bottom25 = dict(q3.id.value_counts() <= q3.id.value_counts().quantile(0.25))
    q3["bottom25"] = q3.id.map(bottom25)
    q3 = q3[q3.bottom25 == True]
    plt.figure(figsize=(10, 10))
    q3.name.value_counts().plot.pie()
    plt.show()
    # plt.figure(figsize=(10,15))
    # q3.id.value_counts().plot.barh()
    # plt.show()
    q3_df = (
        pd.DataFrame(q3.groupby(["name", "id"]).id.count())
        .rename(columns={"id": "count"})
        .reset_index()
    )
    display(q3_df.sort_values(by="count").reset_index(drop=True))


def q4(df):
    program_name = df.program_name.iloc[0]
    suspicious_ip = (
        df[df.name != "Staff"]
        .groupby(["id", "name"])
        .ip.agg([pd.Series.mode, pd.Series.nunique])
    )
    question4 = pd.DataFrame(suspicious_ip)
    question4 = question4.sort_values(by="nunique", ascending=False)
    question4 = question4[
        question4["nunique"] >= question4["nunique"].quantile(0.99)
    ].reset_index()
    display(question4.style.set_caption(f"Q4 {program_name}"))


def q5(ds, full_stack):
    ds_inter = df_union(ds)
    full_stack["ds_access"] = full_stack.page.apply(
        lambda x: True if (any([(x == d) for d in ds_inter])) else False
    )
    full_stack = (
        full_stack.set_index(full_stack.datetime, drop=True)
        .drop(columns="datetime")
        .sort_index()
    )
    by_date = full_stack.groupby(["datetime"]).ds_access.sum().reset_index()

    y = full_stack[full_stack.name != "Staff"].ds_access

    y.resample("14D").sum().plot(
        title="14 day sum of Full-Stack Students accessing Data Science Core Lessons"
    )
    plt.show()


def q6(df):
    question6 = df[(df.lesson == True) & (df.cohort_window == False)]
    dfname = question6.program_name.iloc[0]
    question6 = (
        pd.DataFrame(question6.page.value_counts())
        .reset_index()
        .rename(columns={"page": "count", "index": "lesson"})
    )
    return question6.nlargest(columns="count", n=10).style.set_caption(f"Q6 {dfname}")


def q7(df):
    question7 = df[df.lesson == True].page
    dfname = df.program_name.iloc[0]

    bottom35 = dict(question7.value_counts() <= question7.value_counts().quantile(0.35))
    question7 = pd.DataFrame(question7)
    question7["least_accessed"] = question7.page.map(bottom35)
    question7 = question7[question7.least_accessed == True]
    question7 = question7.page.value_counts()
    question7 = (
        pd.DataFrame(question7)
        .reset_index()
        .nsmallest(columns="page", n=1, keep="all")
        .rename(columns={"page": "count", "index": "lesson"})
        .sort_values(by="lesson")
        .reset_index(drop=True)
    )
    return question7.style.set_caption(f"q7 {dfname}")
