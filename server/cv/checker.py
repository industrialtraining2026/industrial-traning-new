import os
import tempfile
import re
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def check_tesseract_available() -> Tuple[bool, str]:
    """Check if Tesseract OCR is available on the system."""
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    try:
        pytesseract.get_tesseract_version()
        return True, None
    except:
        pass
    
    for path in tesseract_paths:
        if os.path.exists(path):
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                pytesseract.get_tesseract_version()
                return True, path
            except:
                continue
    
    return False, None


def extract_text_with_ocr(pdf_path: str) -> str:
    """Extract text from PDF, using OCR if needed."""
    full_text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += text + "\n"
        
        if len(full_text.strip()) < 100:
            ocr_available, tesseract_path = check_tesseract_available()
            
            if not ocr_available:
                return full_text
            
            try:
                if tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                
                images = convert_from_path(pdf_path, dpi=200)
                ocr_text = ""
                
                for image in images:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    ocr_text += page_text + "\n"
                
                if len(ocr_text.strip()) > len(full_text.strip()):
                    full_text = ocr_text
                    
            except Exception as e:
                logger.warning(f"OCR processing failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
    
    return full_text


def smart_match(text: str, keywords_list: List[str], section_name: str = "") -> bool:
    """Smart matching that handles synonyms and variations."""
    text_lower = text.lower()
    section_lower = section_name.lower()
    
    synonyms = {
        "contact": ["contact", "phone", "email", "address", "mobile", "tel", "e-mail", "mail", "telephone", "whatsapp", "linkedin"],
        "personal": ["personal", "name", "profile", "about me", "about", "bio", "personal information", "personal details"],
        "education": ["education", "academic", "qualification", "degree", "university", "college", "school", "study", "studies", "diploma", "bachelor", "master", "phd", "certificate", "certification"],
        "experience": ["experience", "work", "employment", "job", "career", "professional", "employment history", "work history", "career history", "employment record", "work experience", "working experience", "professional experience", "internship", "intern"],
        "skill": ["skill", "skills", "technical skill", "technical skills", "soft skill", "soft skills", "hard skill", "hard skills", "programming skills", "computer skills", "software skills", "core competencies", "key skills", "professional skills"],
        "activity": ["activity", "activities", "extracurricular", "co-curricular", "achievement", "achievements", "award", "awards", "honor", "honors", "volunteer", "volunteering", "project", "projects", "competition", "competitions"],
        "reference": ["reference", "references", "referee", "referees", "recommendation", "recommendations", "recommended by"]
    }
    
    is_skills_section = "skill" in section_lower or any("skill" in kw.lower() for kw in keywords_list)
    
    if not is_skills_section:
        for keyword in keywords_list:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in text_lower:
                return True
            word_boundary_pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(word_boundary_pattern, text_lower):
                return True
    
    if not is_skills_section and section_lower in synonyms:
        for synonym in synonyms[section_lower]:
            word_boundary_pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(word_boundary_pattern, text_lower):
                return True
    
    if is_skills_section:
        lines = text_lower.split('\n')
        for line in lines:
            line_stripped = line.strip()
            skills_header_patterns = [
                r'^skills?\s*:?\s*$',
                r'^technical\s+skills?\s*:?\s*$',
                r'^soft\s+skills?\s*:?\s*$',
                r'^hard\s+skills?\s*:?\s*$',
                r'^programming\s+skills?\s*:?\s*$',
                r'^computer\s+skills?\s*:?\s*$',
                r'^software\s+skills?\s*:?\s*$',
                r'^core\s+competencies?\s*:?\s*$',
                r'^key\s+skills?\s*:?\s*$',
                r'^professional\s+skills?\s*:?\s*$',
            ]
            for pattern in skills_header_patterns:
                if re.match(pattern, line_stripped):
                    return True
            if re.match(r'^skills?\s*[:•\-]\s*', line_stripped):
                return True
        return False
    
    return False


def check_cgpa(text: str) -> bool:
    """Check if CV contains CGPA/GPA information."""
    text_lower = text.lower()
    
    cgpa_patterns = [
        "cgpa", "gpa", "c.g.p.a", "g.p.a",
        "cumulative grade point average",
        "grade point average"
    ]
    
    for pattern in cgpa_patterns:
        if pattern in text_lower:
            return True
    
    cgpa_numeric_pattern = r'\b(cgpa|gpa|c\.g\.p\.a|g\.p\.a)\s*:?\s*(\d+\.?\d*)\s*/?\s*(\d+\.?\d*)?'
    if re.search(cgpa_numeric_pattern, text_lower):
        return True
    
    gpa_range_pattern = r'\b([2-4]\.\d{1,2})\s*/?\s*([2-5]\.\d{1,2})?'
    education_keywords = ["education", "academic", "degree", "university", "college", "bachelor", "master"]
    if any(keyword in text_lower for keyword in education_keywords):
        matches = re.findall(gpa_range_pattern, text_lower)
        for match in matches:
            value = float(match[0])
            if 2.0 <= value <= 4.5:
                return True
    
    return False


def check_cv(pdf_content: bytes) -> Dict[str, Any]:
    """
    Check an uploaded CV for required sections.
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        Dictionary with check results
    """
    required_section_groups = {
        "Contact Info": [
            "personal details", "contact information", "contact", "phone", "email", 
            "address", "mobile", "tel", "e-mail", "mail", "name", "profile"
        ],
        "Education": [
            "education", "academic", "qualification", "degree", "university", 
            "college", "school", "study", "studies", "diploma", "bachelor", "master"
        ],
        "Experience": [
            "work experience", "working experience", "professional experience",
            "employment", "job", "career", "employment history", "work history",
            "career history", "employment record"
        ],
        "Skills": [
            "skills", "skill", "technical skills", "technical skill", "soft skills", "soft skill",
            "hard skills", "hard skill", "programming skills", "computer skills", "software skills",
            "core competencies", "key skills", "professional skills"
        ],
        "References": [
            "references", "reference", "referee", "referees", 
            "recommendation", "recommendations"
        ]
    }
    
    optional_section_groups = {
        "Activities": [
            "extracurricular activities", "co-curricular activities", "achievements",
            "activity", "activities", "achievement", "award", "awards", 
            "honor", "honors", "volunteer", "volunteering"
        ]
    }
    
    section_requirements = {
        "Contact Info": {
            "title": "Contact Information / Personal Details",
            "should_include": [
                "Full name",
                "Phone number / Mobile number",
                "Email address",
                "Address (optional but recommended)",
                "LinkedIn profile (optional)"
            ],
            "description": "Your contact information so employers can reach you."
        },
        "Education": {
            "title": "Education / Academic Qualifications",
            "should_include": [
                "Degree/Diploma name (e.g., Bachelor of Science, Diploma)",
                "Field of study / Major",
                "University/College name",
                "Graduation date or expected graduation",
                "CGPA/GPA (required, regardless of score)",
                "Relevant coursework (optional)"
            ],
            "description": "Your educational background and qualifications."
        },
        "Experience": {
            "title": "Work Experience / Employment History",
            "should_include": [
                "Job title / Position",
                "Company/Organization name",
                "Employment period (start date - end date)",
                "Job responsibilities / Duties",
                "Key achievements / Accomplishments",
                "Skills used / Technologies involved"
            ],
            "description": "Your work experience, internships, or part-time jobs."
        },
        "Skills": {
            "title": "Skills / Competencies",
            "should_include": [
                "Technical skills (e.g., programming languages, software)",
                "Soft skills (e.g., communication, teamwork, leadership)",
                "Language skills (if applicable)",
                "Certifications (if any)",
                "Proficiency level (e.g., beginner, intermediate, advanced)"
            ],
            "description": "Your relevant skills and competencies."
        },
        "Activities": {
            "title": "Activities / Achievements",
            "should_include": [
                "Extracurricular activities",
                "Volunteer work",
                "Awards and honors",
                "Competitions participated",
                "Projects (academic or personal)",
                "Leadership roles",
                "Clubs or societies involvement"
            ],
            "description": "Your activities, achievements, and involvement outside academics/work."
        },
        "References": {
            "title": "References / Referees",
            "should_include": [
                "Referee name",
                "Referee title/position",
                "Company/Organization",
                "Contact information (email or phone)",
                "Relationship (e.g., former supervisor, professor)"
            ],
            "description": "People who can vouch for your work and character (usually 2-3 references)."
        }
    }
    
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file_path = tmp_file.name
        
        full_text = extract_text_with_ocr(tmp_file_path)
        text_length = len(full_text.strip())
        
        if text_length == 0:
            return {
                "is_complete": False,
                "missing_sections": list(required_section_groups.keys()),
                "missing_optional_sections": list(optional_section_groups.keys()),
                "has_cgpa": False,
                "section_requirements": section_requirements,
                "text_length": 0,
                "error": "Could not extract any text from the PDF."
            }
        
        found_required_sections = set()
        for group_name, keywords in required_section_groups.items():
            if smart_match(full_text, keywords, section_name=group_name):
                found_required_sections.add(group_name)
        
        found_optional_sections = set()
        for group_name, keywords in optional_section_groups.items():
            if smart_match(full_text, keywords, section_name=group_name):
                found_optional_sections.add(group_name)
        
        has_cgpa = check_cgpa(full_text)
        
        all_required_sections = set(required_section_groups.keys())
        missing_sections = list(all_required_sections - found_required_sections)
        
        all_optional_sections = set(optional_section_groups.keys())
        missing_optional_sections = list(all_optional_sections - found_optional_sections)
        
        if not has_cgpa:
            missing_sections.append("CGPA/GPA")
        
        is_complete = not missing_sections and has_cgpa
        
        return {
            "is_complete": is_complete,
            "missing_sections": missing_sections,
            "missing_optional_sections": missing_optional_sections,
            "has_cgpa": has_cgpa,
            "section_requirements": section_requirements,
            "text_length": text_length,
            "found_sections": list(found_required_sections),
            "found_optional_sections": list(found_optional_sections)
        }
        
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        return {
            "is_complete": False,
            "missing_sections": list(required_section_groups.keys()),
            "missing_optional_sections": list(optional_section_groups.keys()),
            "has_cgpa": False,
            "section_requirements": section_requirements,
            "text_length": 0,
            "error": str(e)
        }
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass




