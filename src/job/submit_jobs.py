from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient()
client.submit_job(entrypoint="python my_job.py")
