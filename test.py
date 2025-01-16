import matplotlib.pyplot as plt

# Initialize the diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define positions of tables
positions = {
    "STUDENT": (1, 6),
    "COURSE": (6, 6),
    "ENROLL": (3.5, 4),
    "BOOK_ADOPTION": (6, 2),
    "TEXT": (10, 2)
}

# Table data (headers only for simplicity)
tables = {
    "STUDENT": ["SSN", "Name", "Major"],
    "COURSE": ["Course#", "Cname"],
    "ENROLL": ["SSN", "Course#", "Quarter"],
    "BOOK_ADOPTION": ["Course#", "Quarter", "Book_ISBN"],
    "TEXT": ["Book_ISBN", "Book_Title", "Publisher"]
}

# Draw the tables
for table, (x, y) in positions.items():
    header = f"{table}\n" + "\n".join(tables[table])
    ax.text(x, y, header, fontsize=10, bbox=dict(boxstyle="round", facecolor="lightblue"), ha='center')

# Draw arrows for foreign keys
arrows = [
    ("ENROLL", "STUDENT", (3.5, 4.5), (1, 6.5)),  # ENROLL → STUDENT
    ("ENROLL", "COURSE", (3.5, 4.5), (6, 6.5)),  # ENROLL → COURSE
    ("BOOK_ADOPTION", "COURSE", (6, 2.5), (6, 5.5)),  # BOOK_ADOPTION → COURSE
    ("BOOK_ADOPTION", "TEXT", (7.5, 2), (9, 2))  # BOOK_ADOPTION → TEXT
]

for start, end, pos_start, pos_end in arrows:
    ax.annotate(
        "", xy=pos_end, xycoords="data",
        xytext=pos_start, textcoords="data",
        arrowprops=dict(arrowstyle="<|-", color="black", lw=1.5)
    )

plt.title("Relational Schema Diagram with Foreign Keys", fontsize=14)
plt.show()
