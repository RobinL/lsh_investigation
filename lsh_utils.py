import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as f
from  pyspark.ml.feature import *

def get_spark():
    
    
    
    conf=SparkConf()
    conf.set('spark.driver.memory', '13g')
    conf.set("spark.sql.shuffle.partitions", "40")
    conf.set('spark.driver.extraClassPath', 'jars/scala-udf-similarity-0.0.6.jar')
    conf.set('spark.jars', 'jars/scala-udf-similarity-0.0.6.jar')    
    sc = SparkContext.getOrCreate(conf=conf)


    spark = SparkSession(sc)
    


    # Register UDFs
    from pyspark.sql import types

    udfs = [
        ('jaro_winkler_sim', 'JaroWinklerSimilarity',types.DoubleType()),
    ('jaccard_sim', 'JaccardSimilarity',types.DoubleType()),
    ('cosine_distance', 'CosineDistance',types.DoubleType()),
    ('Dmetaphone', 'DoubleMetaphone',types.StringType()),
    ('QgramTokeniser', 'QgramTokeniser',types.StringType()),
    ('Q3gramTokeniser','Q3gramTokeniser',types.StringType()),
    ('Q4gramTokeniser','Q4gramTokeniser',types.StringType()),
    ('Q5gramTokeniser','Q5gramTokeniser',types.StringType())
    ]

    for a,b,c in udfs:
        spark.udf.registerJavaFunction(a, 'uk.gov.moj.dash.linkage.'+ b, c)
    
    return spark

def get_and_process_fake_data(path, spark):
    df = spark.read.parquet(path)
    df.createOrReplaceTempView("df")
    sql = """
    select *, concat_ws(" ", first_name, surname, dob, city) as concat
    from 
    df
    """
    df = spark.sql(sql)
    df.createOrReplaceTempView("df")


    df = df.withColumn("concat", f.regexp_replace(df["concat"], "19(\d\d)", "$1"))
    df = df.withColumn("concat", f.regexp_replace(df['concat'], "\s{2,10}", " "))
    df = df.withColumn("concat", f.regexp_replace(df["concat"], "\-", ""))
    df = df.withColumn("concat", f.lower(df['concat']))
    return df

def get_qgrams(df, spark):
    
    df.createOrReplaceTempView("df")
#     sql = """
#     select unique_id, concat, 
#     split(Q5gramTokeniser(concat), ' ') as qgram5, 
#     split(Q4gramTokeniser(concat), ' ') as qgram4, 
#     split(Q3gramTokeniser(concat), ' ') as qgram3, 
#     split(QgramTokeniser(concat), ' ') as qgram2
#     from df
#     """
    sql = """
    select unique_id, concat, group, split(Q3gramTokeniser(concat), ' ') as qgrams
    from df
    """
    df_q = spark.sql(sql)
    
#     df_q.createOrReplaceTempView("df_q")
#     sql = """
#     select unique_id, concat, array_union(array_union(array_union(qgram5, qgram4), qgram3), qgram2) as qgrams
#     from df_q
#     """
#     df_q = spark.sql(sql)
    df_q.createOrReplaceTempView("df_q")
    return df_q



from pyspark.sql.types import *
from pyspark.sql.functions import lit, udf

def get_size_(v):
    try:
        return v.numNonzeros()
    except ValueError:
        return None
    

def get_features(feature_type, df_q, settings, spark):
    
    if feature_type == "countvectorise":
        cv = CountVectorizer(inputCol="qgrams", outputCol="features", **settings)
        model = cv.fit(df_q)
        result = model.transform(df_q)
        
        # Get rid of records where countvectoriser is null
        get_size = udf(get_size_, LongType())
        spark.udf.register("get_size", get_size)
        
        result.createOrReplaceTempView("result")
        sql = """
        select *
        from result
        where get_size(features) > 0
        """
        result = spark.sql(sql)
        return result
    
    if feature_type == "cvidf":
        cv = CountVectorizer(inputCol="qgrams", outputCol="rawFeatures", **settings)
        model = cv.fit(df_q)
        result = model.transform(df_q)
        
        # Get rid of records where countvectoriser is null
        get_size = udf(get_size_, LongType())
        spark.udf.register("get_size", get_size)
        
        result.createOrReplaceTempView("result")
        sql = """
        select *
        from result
        where get_size(rawFeatures) > 0
        """
        result = spark.sql(sql)
        
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(result)
        return idfModel.transform(result)
        
    
    if feature_type == "hashingtf":
        hashingTF = HashingTF(inputCol="qgrams", outputCol="features", **settings)
        return hashingTF.transform(df_q)
    
    if feature_type == "tfidf":
        hashingTF = HashingTF(inputCol="qgrams", outputCol="rawFeatures", **settings)
        featurizedData = hashingTF.transform(df_q)


        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(featurizedData)
        return idfModel.transform(featurizedData)
        
        
def get_lsh_model(num_hash_tables, features, spark):
    
    mh = MinHashLSH().setNumHashTables(num_hash_tables).setInputCol("features").setOutputCol("hashValues")
    return mh.fit(features)
    
    


def get_from_hash_value_(v):
    try:
        return int(v[0])
    except ValueError:
        return None

def get_hash_groups(num_hash_tables, hash_values, spark):
    hash_values.createOrReplaceTempView("hash_values")
    get_from_hash_value = udf(get_from_hash_value_, LongType())
    spark.udf.register("get_from_hash_value", get_from_hash_value_)


    tem = "get_from_hash_value(element_at(hashValues,{t})) as hash{t}"
    sel = [tem.format(t=t+1) for t in range(num_hash_tables)]


    sel = ", ".join(sel)

    
    sql = f"""
    select unique_id, {sel}
    from hash_values

    """
    split_hash = spark.sql(sql)
    split_hash.createOrReplaceTempView("split_hash")

    tem = """
    select hash{t} as hash, count(*) as count_hash, '{t}' as hashtable
    from split_hash
    group by hash{t}
    """
    tabs = [tem.format(t=t+1) for t in range(num_hash_tables)]

    tabs = "\n union all \n".join(tabs)

    sql = f"""
    {tabs}

    order by count_hash desc


    """
    hash_groups = spark.sql(sql)
    return hash_groups

def get_num_comparisons(hash_groups, spark):
    
    hash_groups.createOrReplaceTempView("hash_groups")

    sql = """
    select sum(count_hash*count_hash) as sum_of_comparisons
    from hash_groups
    """

    total_df = spark.sql(sql)
    return total_df.collect()[0]["sum_of_comparisons"]

def get_comparisons(lsh_model, features, threshold, spark):
    
    approx_join = lsh_model.approxSimilarityJoin(features, features, threshold, distCol="distance")
    
    approx_join.createOrReplaceTempView("approx_join")
    sql = """
    select 
    datasetA.concat as concat_l,
    datasetA.group as group_l,
    datasetA.unique_id as unique_id_l, 
    datasetB.concat as concat_r, 
    datasetB.group as group_r,
    datasetB.unique_id as unique_id_r, 
    distance
    from approx_join
    where datasetA.unique_id < datasetB.unique_id
    """
    return spark.sql(sql)

def get_accuracy_stats(df, comparisons, num_actual_comparisons, spark):
    
    c = df.count()
    total_comparisons = c * (c-1) / 2
    
    
    df.createOrReplaceTempView("df")
    sql = """
    select sum(num_positives) as num_positives
    from (
    select group, count(*)*(count(*)-1)/2 as num_positives
    from df
    group by group)

    """
    total_positives = spark.sql(sql).collect()[0]["num_positives"]

    comparisons.createOrReplaceTempView("comparisons")
    sql = """

    select classification, cast(count(*) as double) as class_count
    from (
    select 
    CASE
    WHEN group_l = group_r THEN 'tp'
    WHEN group_l != group_r THEN 'fp'
    END as classification
    from comparisons
    )
    group by classification


    """
    cc =spark.sql(sql).toPandas().set_index("classification").to_dict()["class_count"]
    if "tp" not in cc:
        cc["tp"] = 0
    if "fp" not in cc:
        cc["fp"] = 0
    
    cc["fn"] = total_positives - cc["tp"]
    cc["tn"] = total_comparisons - cc["tp"] - cc["fp"] - cc["fn"]

    cc["sensitivity"] = cc["tp"]/(cc["tp"]+cc["fn"])
    cc["precision"] = cc["tp"]/(cc["tp"]+cc["fp"])
    
    cc["total_potential_comparisons"] = total_comparisons
    cc["num_actual_comparisons"] = num_actual_comparisons
    
    
    return cc

def generate_report(accuracy_stats):
    cc = accuracy_stats
    report = f"""
    You're making {cc["num_actual_comparisons"]:,.0f} from a total of {cc["total_potential_comparisons"]:,.0f} comparisons
    That's 1 comparison for every {cc["total_potential_comparisons"]/cc["num_actual_comparisons"]:,.1f} potential comparisons

    Sensitivity = {cc["sensitivity"]:,.0%}.  i.e. you're missing {1-cc["sensitivity"]:,.0%} of actual matches
    Precision = {cc["precision"]:,.0%} i.e. {1-cc["precision"]:,.0%} of comparisons are not matches

    True positives  = {cc["tp"]:,.0f}
    False positives = {cc["fp"]:,.0f}
    True negatives  = {cc["tn"]:,.0f}
    False negatives = {cc["fn"]:,.0f}

    """
    return report

def run_all(path_to_data, num_hash_tables, feature_type, settings, threshold, spark):
    df = get_and_process_fake_data(path_to_data, spark)
    df.persist()
    df_q = get_qgrams(df,spark)
    features = get_features(feature_type, df_q, settings, spark)
    lsh_model = get_lsh_model(num_hash_tables, features, spark)
    hash_values = lsh_model.transform(features)
    hash_groups = get_hash_groups(num_hash_tables, hash_values, spark)
    hash_groups.persist()
    top_groups = list(hash_groups.limit(10).toPandas()["count_hash"])
    num_actual_comparisons = get_num_comparisons(hash_groups, spark)
    c = df.count()
    total_comparisons = c * (c-1) / 2
                                                 
    if total_comparisons/num_actual_comparisons < 5:
        return {"total_potential_comparisons": total_comparisons,
                "num_actual_comparisons": num_actual_comparisons}
                                                 
    hash_groups.unpersist()
    
    
    
    comparisons = get_comparisons(lsh_model, features, threshold, spark)
    
    accuracy_stats = get_accuracy_stats(df, comparisons, num_actual_comparisons, spark)
    
    
    df.unpersist()
    
    return accuracy_stats
    