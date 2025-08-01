from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from course_pipeline import CoursePipeline
from reco_cours.data.course_classifier import CourseClassifier, CATEGORIES

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Course Pipeline API",
    description="API pour la classification et l'enrichissement des cours",
    version="1.0.0"
)

# Modèles Pydantic pour la validation des données
class CourseContent(BaseModel):
    paragraphs: List[str] = []
    lists: Optional[List[List[str]]] = None
    examples: Optional[List[str]] = None

class CourseInput(BaseModel):
    titre: str
    description: str
    contenus: CourseContent

class CourseOutput(BaseModel):
    titre: str
    description: str
    contenus: CourseContent
    categorie: str
    score_categorie: float
    niveau: int
    probabilites_niveau: Dict[str, float]

# Initialisation du pipeline
MODEL_PATH = "./level_model.joblib"
VECTORIZER_PATH = "./vectorizer.joblib"

# Création d'une instance globale du pipeline
categories_classifier = CourseClassifier(CATEGORIES)
pipeline = CoursePipeline(
    categories_classifier=categories_classifier,
    level_model_path=MODEL_PATH,
    vectorizer_path=VECTORIZER_PATH
)

@app.post("/process_course", response_model=CourseOutput)
async def process_single_course(course: CourseInput):
    """
    Traite un cours unique pour prédire sa catégorie et son niveau.
    """
    try:
        course_dict = course.dict()
        result = pipeline.process_single_course(course_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_courses_batch", response_model=List[CourseOutput])
async def process_courses_batch(courses: List[CourseInput]):
    """
    Traite un lot de cours pour prédire leurs catégories et niveaux.
    """ 
    try:
        courses_dict = [course.dict() for course in courses]
        results = pipeline.process_courses_batch(courses_dict)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_file")
async def process_file(input_file: str, output_file: str, batch_size: int = 32):
    """
    Traite un fichier JSON contenant des cours.
    """
    try:
        input_path = Path(input_file)
        output_path = Path(output_file)
        pipeline.process_file(input_path, output_path, batch_size)
        return {"message": f"Fichier traité avec succès. Résultats sauvegardés dans {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 