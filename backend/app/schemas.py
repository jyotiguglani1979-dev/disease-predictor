from pydantic import BaseModel, Field

class AuthRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=10, max_length=200)

class StartAssessment(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: str = Field(pattern="^(Female|Male|Other / not specified|M|F)$")
    evidence: list[str] = Field(default_factory=list, max_length=100)
    free_text: str = Field(default="", max_length=3000)
    consent: bool = False
    allow_gemini: bool = False

class AnswerRequest(BaseModel):
    evidence: str | None = Field(default=None, max_length=100)
    code: str | None = Field(default=None, max_length=32)
    response: str = Field(default="yes", pattern="^(yes|no|unsure)$")
    allow_gemini: bool = False

class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    allow_gemini: bool = False

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    assessment_id: str | None = None
    allow_gemini: bool = False

class ExplanationRequest(BaseModel):
    allow_gemini: bool = False
    free_text: str = Field(default="", max_length=3000)

class HeartInput(BaseModel):
    age: float = Field(ge=18, le=110)
    sex: str
    current_smoker: int = Field(ge=0, le=1)
    systolic_bp: float = Field(ge=70, le=260)
    bp_treatment: int = Field(ge=0, le=1)
    total_cholesterol: float = Field(ge=80, le=700)
    hdl: float = Field(ge=10, le=200)
    diabetes: int = Field(ge=0, le=1)
    consent: bool = False
