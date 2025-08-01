def order_courses(courses_with_scores):
    # Trier par niveau croissant, puis score décroissant
    return sorted(
        courses_with_scores, 
        key=lambda x: (x[0].get("niveau", 0), -x[1])
    )
