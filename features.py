#!/usr/bin/env python3

"""
Feature extraction module for DualSpace Representative Selection (DSRS)

Provides:

1. Semantic embeddings
2. Structural question representations

"""

import numpy as np

import spacy

from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import StandardScaler


###############################################################################
# Semantic Features
###############################################################################

def compute_semantic_embeddings(
    questions,
    model_name="all-MiniLM-L6-v2",
):
    """
    Compute semantic embeddings.

    Parameters
    ----------
    questions : list[str]

    model_name : str

    Returns
    -------
    numpy.ndarray

    Shape:
        (number_of_questions, embedding_dimension)

    """

    print(
        f"Loading embedding model: {model_name}"
    )

    model = SentenceTransformer(model_name)


    embeddings = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


    return np.asarray(embeddings)



###############################################################################
# Structural Features
###############################################################################


def safe_div(a, b):

    if b == 0:
        return 0.0

    return a / b



def dependency_depth(doc):
    """
    Maximum dependency tree depth.
    """

    depths = []


    for token in doc:

        depth = 0

        current = token


        while current.head != current:

            depth += 1

            current = current.head


        depths.append(depth)


    if len(depths) == 0:
        return 0


    return max(depths)



def average_dependency_depth(doc):

    depths = []


    for token in doc:

        depth = 0

        current = token


        while current.head != current:

            depth += 1

            current = current.head


        depths.append(depth)


    if len(depths) == 0:
        return 0


    return np.mean(depths)



def question_type_features(doc):

    """
    Encode question form.

    WH words are important because
    different RAG failures occur for
    different question types.
    """

    wh_words = [
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    ]


    text = doc.text.lower()


    return [
        int(text.startswith(w))
        for w in wh_words
    ]



def pos_distribution(doc):

    """
    POS histogram.

    Preserves syntactic style.
    """

    tags = [

        "NOUN",
        "VERB",
        "ADJ",
        "ADV",
        "PRON",
        "DET",
        "ADP",
        "NUM",
        "PROPN",
        "AUX",

    ]


    counts = {
        x:0
        for x in tags
    }


    for token in doc:

        if token.pos_ in counts:

            counts[token.pos_] += 1



    total = len(doc)


    return [
        safe_div(
            counts[x],
            total
        )
        for x in tags
    ]



def entity_features(doc):

    """
    Entity statistics.

    """

    entity_types = [

        "PERSON",
        "ORG",
        "GPE",
        "DATE",
        "TIME",
        "CARDINAL",
        "EVENT",

    ]


    counts = {
        x:0
        for x in entity_types
    }


    for ent in doc.ents:

        if ent.label_ in counts:

            counts[ent.label_] += 1



    total_entities = len(doc.ents)


    features = [

        total_entities,

    ]


    for x in entity_types:

        features.append(
            counts[x]
        )


    return features



def lexical_features(doc):

    """
    Basic lexical properties.
    """

    tokens = [

        token.text
        for token in doc
        if not token.is_punct

    ]


    lengths = [

        len(t)
        for t in tokens

    ]


    if len(lengths)==0:

        return [
            0,
            0,
            0,
        ]


    return [

        len(tokens),

        np.mean(lengths),

        np.std(lengths),

    ]



def punctuation_features(doc):

    """

    Punctuation style.

    """

    question_marks = 0

    commas = 0


    for token in doc:

        if token.text == "?":

            question_marks += 1


        if token.text == ",":

            commas += 1


    return [

        question_marks,

        commas,

    ]



def numeric_features(doc):

    """

    Numerical reasoning indicator.

    """

    numbers = 0


    for token in doc:

        if token.like_num:

            numbers += 1


    return [

        numbers,

        safe_div(
            numbers,
            len(doc)
        ),

    ]



###############################################################################
# Main Structural Extraction
###############################################################################


def compute_structural_features(
    questions,
    spacy_model="en_core_web_sm",
):
    """
    Convert questions into structural vectors.

    Returns:

        numpy array

        shape:

        (N, structural_dimension)

    """


    print(
        f"Loading spaCy model: {spacy_model}"
    )


    nlp = spacy.load(
        spacy_model
    )


    feature_vectors = []


    print(
        "Extracting structural features..."
    )


    for question in tqdm(questions):


        doc = nlp(question)


        features = []


        #
        # lexical
        #

        features.extend(
            lexical_features(doc)
        )


        #
        # dependency structure
        #

        features.extend(
            [
                dependency_depth(doc),
                average_dependency_depth(doc),
            ]
        )


        #
        # POS distribution
        #

        features.extend(
            pos_distribution(doc)
        )


        #
        # Entities
        #

        features.extend(
            entity_features(doc)
        )


        #
        # Question type
        #

        features.extend(
            question_type_features(doc)
        )


        #
        # punctuation
        #

        features.extend(
            punctuation_features(doc)
        )


        #
        # numbers
        #

        features.extend(
            numeric_features(doc)
        )


        feature_vectors.append(
            features
        )



    X = np.asarray(
        feature_vectors,
        dtype=float
    )


    #
    # Normalize structural space
    #

    scaler = StandardScaler()

    X = scaler.fit_transform(X)


    return X