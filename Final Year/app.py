import gradio as gr
from utils import make_predictions
iface = gr.Interface(
    fn=make_predictions,
    inputs=[
        gr.Number(label="Number of dependents",info="Number of dependents i.e. family members",),
        gr.Radio(label="Education", choices=["Graduate", "Not Graduate"],info="Education of the applicant",value="Graduate",),
        gr.Radio(label="Self Employed", choices=["Yes", "No"],info="Is the applicant self employed?",value="Yes",),
        gr.Number(label="Annual Income",info="Annual income of the applicant with Minimum Requirement of 100000", minimum=100000,step=1000,value=100000,),
        gr.Number(label="Loan Term",info="Loan term in months minimum 3 ",minimum=3,maximum=24,value=3,),
        gr.Number(label="CIBIL Score",info="CIBIL Score of the applicant and minimum is 300 maximum is 800",minimum=300,maximum=800,value=300,),
        gr.Number(label="Residential Assets Value",info="Value of residential assets of the applicant Example:House,Land"),
        gr.Number(label="Commercial Assets Value",info="Value of commercial assets of the applicant Example:Shop,Shares,Factory"),
        gr.Number(label="Luxury Assets Value",info="Value of luxury assets of the applicant Example:Cars,Private Vehicles"),
        gr.Number(label="Bank Asset Value",info="Value of bank assets of the applicant Example:Bank Balance"),
    ],
    
    outputs=[gr.Label(label= "Please Press Flag If error",show_label=True)],
    theme="xiaobaiyuan/theme_brief",
    live=False,
    title="Loan Approval Prediction System",
    description="This system predicts whether a loan will be approved or rejected based on the user inputs.",
    examples=[
        [2, "Graduate", "Yes", 150000, 12, 600, 200000, 0, 50000, 100000],
        [1, "Not Graduate", "No", 80000, 6, 400, 100000, 50000, 0, 50000],
        [3, "Graduate", "No", 200000, 24, 750, 300000, 100000, 100000, 200000],
    ]
)

iface.launch(debug=True, share=True,server_port=7860)
