"""
Import repository
"""
from src.repo.teacher_repo import TeacherRepo
from src.repo.student_repo import StudentRepo


class AuthService:
    """
    This class implement for auth
    """

    def __init__(
            self,
            teacher_repo: TeacherRepo,
            student_repo: StudentRepo):
        self.teacher_repo = teacher_repo
        self.student_repo = student_repo

    def login(self, email: str, password: str):
        """
        This function handles the login process and returns the result in JSON format.
        """
        teacher = self.teacher_repo.get_teacher_by_email(email, password)
        if teacher:
            return teacher

        student = self.student_repo.get_student_by_email(email, password)
        if student:
            return student

        return None
