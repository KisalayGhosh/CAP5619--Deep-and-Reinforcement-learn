import xml.etree.ElementTree as ET
import pandas as pd

def parse_absa_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text

        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect in aspect_terms.findall('aspectTerm'):
                term = aspect.attrib['term']
                polarity = aspect.attrib['polarity']

                if polarity not in ['positive', 'neutral', 'negative']:
                    continue  

                data.append({
                    'sentence': text,
                    'aspect': term,
                    'sentiment': polarity
                })

    return pd.DataFrame(data)


df = parse_absa_xml("Restaurants_Train.xml")
print(df.head())
df.to_csv("absa_restaurants.csv", index=False)
