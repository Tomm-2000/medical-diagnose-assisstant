from py2neo import Graph
from py2neo.errors import ConnectionUnavailable, ServiceUnavailable
import time

def kg_summary():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "password"))
            break
        except (ConnectionUnavailable, ServiceUnavailable) as e:
            if attempt == max_retries - 1:
                print(f" فشل الاتصال بعد {max_retries} محاولات: {e}")
                return
            print(f" محاولة الاتصال {attempt+1} فشلت، إعادة المحاولة بعد 5 ثوان...")
            time.sleep(5)

    print("=== Knowledge Graph Summary ===\n")

    # 1. عدد العقد حسب النوع
    result = graph.run("MATCH (n) RETURN labels(n) AS type, count(*) AS count")
    print("Node Counts by Type:")
    for record in result:
        print(f"  {record['type'][0]}: {record['count']}")

    # 2. عدد العلاقات حسب النوع
    result = graph.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
    print("\nRelationship Counts by Type:")
    for record in result:
        print(f"  {record['type']}: {record['count']}")

    # 3. أكثر 10 أمراض
    result = graph.run("""
        MATCH (d:Disease)<-[:DISCUSSES]-(a:Article)
        RETURN d.name AS disease, count(a) AS articles
        ORDER BY articles DESC LIMIT 10
    """)
    print("\nTop 10 Diseases by Article Count:")
    for record in result:
        print(f"  {record['disease']}: {record['articles']} articles")

    # 4. أكثر 10 أعراض
    result = graph.run("""
        MATCH (s:Symptom)<-[:MENTIONS]-(a:Article)
        RETURN s.name AS symptom, count(a) AS mentions
        ORDER BY mentions DESC LIMIT 10
    """)
    print("\nTop 10 Symptoms by Mention Count:")
    for record in result:
        print(f"  {record['symptom']}: {record['mentions']} mentions")

if __name__ == "__main__":
    kg_summary()
