import pandas as pd
from jobspy import scrape_jobs

def fetch_live_jobs(search_term: str, location: str, max_jobs: int = 5):
    """
    Scrapes LinkedIn and Indeed for live job postings.
    Bypasses basic bot detection automatically.
    """
    print(f"🕵️‍♂️ AGENT ACTIVE: Searching for '{search_term}' in {location}...")
    
    try:
        # jobspy handles the heavy lifting of concurrent scraping
        jobs_df = scrape_jobs(
            site_name=["linkedin", "indeed"],
            search_term=search_term,
            location=location,
            results_wanted=max_jobs,
            country_indeed='INDIA' # Adjust based on your target region
        )
        
        if jobs_df.empty:
            print("❌ No jobs found. Try broadening your search.")
            return []

        # We only need specific columns for our AI to analyze
        # Format: Title, Company, Job URL, and the full Description
        selected_columns = ["title", "company", "job_url", "description"]
        clean_jobs = jobs_df[selected_columns].dropna()
        
        # Convert the Pandas DataFrame into a list of dictionaries for our agent
        job_list = clean_jobs.to_dict(orient='records')
        
        print(f"✅ Successfully harvested {len(job_list)} live job postings.")
        return job_list
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return []

if __name__ == "__main__":
    # Test the scraper
    print("=== TESTING JOB HARVESTER ===")
    
    # Let's search for something that aligns with your actual resume
    target_role = "Linux System Administrator"
    target_location = "Bengaluru"
    
    live_jobs = fetch_live_jobs(target_role, target_location, max_jobs=5)
    
    for i, job in enumerate(live_jobs):
        print(f"\n--- JOB {i+1} ---")
        print(f"Title:   {job['title']}")
        print(f"Company: {job['company']}")
        print(f"URL:     {job['job_url']}")
        # Print just the first 150 characters of the description to verify it worked
        print(f"Desc:    {job['description'][:500]}...")