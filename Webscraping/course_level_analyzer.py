import json
import spacy
from typing import Dict, List
import re

class ProgrammingCourseLevelAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')        
        # Programming keywords by level
        self.level_keywords = {
            'Beginner': [
                # Basic concepts
                'introduction', 'getting started', 'begin', 'start',
                'basics', 'fundamentals', 'initiation',
                # Basic structures
                'variables', 'types', 'loops', 'conditions', 'if else',
                'basic functions', 'print', 'hello world',
                # Languages often used for beginners
                'scratch', 'python for beginners', 'html css'
            ],
            
            'Intermediate': [
                # Intermediate concepts
                'object oriented', 'classes', 'objects', 'inheritance',
                'api', 'rest', 'json', 'xml', 'databases', 'sql',
                'frameworks', 'git', 'unit testing',
                # Advanced structures
                'collections', 'arrays', 'lists', 'dictionaries',
                'exceptions', 'error handling', 'debug',
                # Basic patterns
                'mvc', 'crud'
            ],
            
            'Advanced': [
                # Advanced concepts
                'architecture', 'design patterns', 'solid', 'clean code',
                'microservices', 'cloud', 'devops', 'ci/cd',
                'security', 'cryptography', 'optimization',
                # Complex topics
                'concurrency', 'threading', 'async', 'performance',
                'scalability', 'kubernetes', 'docker',
                'machine learning', 'artificial intelligence',
                # Advanced patterns
                'dependency injection', 'inversion of control',
                'reactive programming', 'event sourcing'
            ]
        }
        
        # Technologies and their typical level
        self.tech_levels = {
            'Beginner': [
                'html', 'css', 'basic javascript', 'python', 'scratch',
                'visual basic', 'basic sql'
            ],
            'Intermediate': [
                'react', 'angular', 'vue', 'node.js', 'express',
                'django', 'flask', 'spring', 'bootstrap',
                'jquery', 'typescript', 'java', 'c#'
            ],
            'Advanced': [
                'kubernetes', 'docker', 'terraform', 'aws', 'azure',
                'graphql', 'elasticsearch', 'kafka', 'hadoop',
                'spark', 'tensorflow', 'pytorch'
            ]
        }

    def analyze_code_complexity(self, text: str) -> float:
        """
        Analyzes the complexity of code mentioned in the course
        """
        # Search for complex code patterns
        complex_patterns = [
            r'class\s+\w+',           # Class definitions
            r'interface\s+\w+',       # Interfaces
            r'async|await',           # Async code
            r'try\s*{.*?}',          # Exception handling
            r'@\w+',                  # Decorators/Annotations
            r'import\s+.*?from',      # Complex imports
            r'extends|implements',    # Inheritance
            r'private|protected',     # Access modifiers
            r'static|final',          # Modifiers
            r'<.*?>'                  # Generics
        ]
        
        complexity_score = 0
        for pattern in complex_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            complexity_score += len(matches) * 0.1
        
        return min(1.0, complexity_score)

    def determine_level(self, course: Dict) -> str:
        """
        Determines the programming course level
        """
        # Get the course data from the correct structure
        course_data = course.get('cours', {})
        title = course_data.get('titre', '')
        description = course_data.get('description', '')
        
        # Get content from paragraphs, lists, and examples
        contenus = course_data.get('contenus', {})
        paragraphs = contenus.get('paragraphs', [])
        lists = contenus.get('lists', [])
        examples = contenus.get('examples', [])
        
        # Combine all text content
        text_content = [title, description]
        text_content.extend(paragraphs)
        # Flatten lists if they exist
        for list_items in lists:
            if isinstance(list_items, list):
                text_content.extend(list_items)
        text_content.extend(examples)
        
        # Join all text together
        text = ' '.join(text_content).lower()
        
        # Rest of your level determination logic
        beginner_keywords = ['basic', 'beginner', 'start', 'introduction', 'fundamental']
        intermediate_keywords = ['intermediate', 'advanced', 'complex', 'deeper', 'optimization']
        advanced_keywords = ['expert', 'professional', 'mastery', 'architecture', 'design patterns']
        
        # Count keyword occurrences
        beginner_count = sum(1 for keyword in beginner_keywords if keyword in text)
        intermediate_count = sum(1 for keyword in intermediate_keywords if keyword in text)
        advanced_count = sum(1 for keyword in advanced_keywords if keyword in text)
        
        # Determine level based on keyword counts
        if advanced_count > intermediate_count and advanced_count > beginner_count:
            return "Advanced"
        elif intermediate_count > beginner_count:
            return "Intermediate"
        else:
            return "Beginner"

    def process_courses(self, input_file: str, output_file: str):
        """
        Process all courses from JSON file and add levels
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for course_data in data:
            course = course_data['cours']
            level = self.determine_level(course)
            course['niveau'] = level
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# Usage example
if __name__ == "__main__":
    import os
    
    analyzer = ProgrammingCourseLevelAnalyzer()
    
    # Create level folder if it doesn't exist
    if not os.path.exists('level'):
        os.makedirs('level')
        
    # Automatically get JSON files from formatted folder
    formatted_dir = 'formatted'
    for filename in os.listdir(formatted_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(formatted_dir, filename)
            output_file = os.path.join('level', filename)
            
            # Analyze and add levels
            analyzer.process_courses(input_file, output_file)