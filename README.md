# Multimodal intent classification with incomplete modalities using text embedding propagation

Determining the author’s intent in a social media post is a challeng-ing multimodal task and requires identifying complex relationshipsbetween image and text in the post. For example, the post imagecan represent an object, person, product, or company, while thetext can be an ironic message about the image content. Similarly, atext can be a news headline, while the image represents a provoca-tion, meme, or satire about the news. Existing approaches proposeintent classification techniques combining both modalities. However, some posts may have missing textual annotations. Hence, we investigate a graph-based approach that propagates available textembedding data from complete multimodal posts to incompleteones. This paper presents a text embedding propagation method,which transfers embeddings from BERT neural language modelsto image-only posts (i.e., posts with incomplete modality) considering the topology of a graph constructed from both visual andtextual modalities available during the training step. By using this inference approach, our method provides competitive results whentextual modality is available at different completeness levels, evencompared to reference methods that require complete modalities.

### Getting the Intent dataset

The provided text embeddings were obtained from a fine-tuned model for the problem, based on: https://huggingface.co/bert-base-uncased

```
$ git clone https://github.com/karansikka1/documentIntent_emnlp19/ ./dataset/
$ wget https://www.dropbox.com/s/pp1nkipzklrgqwl/paper-intent.zip
$ tar -xvf ./dataset/resnet18_feat.tar
$ unzip paper-intent.zip -d ./features
```

### Running the method for Intent classification

Requirements:

```
sklearn == 0.24.*
numpy == 1.21.*
pandas == 1.3.*
```

Running after dataset extraction, as explained above:

```
$ python3 run.py
```

### Citing us:
```
@INPROCEEDINGS{215905,
    AUTHOR="Victor Machado Gonzaga and Nils Murrugarra-Llerena and Ricardo Marcacini",
    TITLE="Multimodal intent classification with incomplete modalities using text embedding propagation",
    BOOKTITLE="WebMedia 2021 () ",
    ADDRESS="",
    DAYS="5-12",
    MONTH="nov",
    YEAR="2021",
    ABSTRACT="Determining the author's intent in a social media post is a challenging multimodal task and requires identifying complex relationships between image and text in the post. For example, the post image can represent an object, person, product, or company, while the text can be an ironic message about the image content. Similarly, a text can be a news headline, while the image represents a provocation, meme, or satire about the news. Existing approaches propose intent classification techniques combining both modalities. However, some posts may have missing textual annotations. Hence, we investigate a graph-based approach that propagates available text embedding data from complete multimodal posts to incomplete multimodal ones. This paper presents a text embedding propagation method, which transfers embeddings from BERT neural language models to image-only posts (i.e., posts with incomplete modality) considering the topology of a graph constructed from both visual and textual modalities available during the training step. By using this inference approach, our method provides competitive results when textual modality is available at different completeness levels, even compared to reference methods that require complete modalities.",
    KEYWORDS="AI, Machine Learning and Deep Learning; Data Mining; Multimodal Interaction",
    URL="http://XXXXX/215905.pdf"
}
```
