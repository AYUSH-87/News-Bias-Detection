"""
Starter Data and GloVe Embedding Generator.

Creates:
1. data/news_bias_dataset.csv: Realistic journalistic opinion dataset balanced across Liberal, Conservative, Neutral stances.
2. data/embeddings/glove_mini.50d.txt: Coherent 50-dimensional starter GloVe embeddings covering extensive political, economic, social, and journalistic vocabulary.
3. data/sample_article.txt: Sample opinion piece for CLI file prediction testing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Define article records: (text, bias)
ARTICLES = [
    # --- HEALTHCARE (Liberal, Conservative, Neutral) ---
    (
        "Access to comprehensive healthcare must be recognized as a fundamental human right rather than a market commodity. Millions of working families are crushed by predatory private insurance premiums and unaffordable prescription drugs. Implementing a single-payer Medicare for All system will eliminate bureaucratic waste, guarantee coverage regardless of employment status, and prioritize patient well-being over corporate pharmaceutical profits.",
        "liberal"
    ),
    (
        "A market-oriented approach to healthcare reform empowers patients through competition and price transparency. Government-run healthcare schemes inevitably result in rationed care, stifled medical innovation, and lengthy waitlists. Expanding Health Savings Accounts and permitting cross-state insurance competition will bring down medical costs without imposing crushing tax burdens on middle-class taxpayers.",
        "conservative"
    ),
    (
        "The Congressional Budget Office published its latest analysis comparing public health options with private exchange subsidies. The report outlines that while universal coverage options reduce uninsured rates by an estimated twelve percent, aggregate federal outlays would increase over the ten-year budget window. Lawmakers continue debating the trade-offs between deficit impact and expanded Medicaid eligibility.",
        "neutral"
    ),
    (
        "Corporate greed in the healthcare sector has reached crisis levels as drug manufacturers artificially inflate life-saving insulin prices to reward Wall Street shareholders. Strong government regulation, aggressive antitrust enforcement against hospital monopolies, and capping out-of-pocket costs are urgent moral imperatives to defend vulnerable patients.",
        "liberal"
    ),
    (
        "Federal mandates and excessive bureaucratic regulations are driving independent physicians out of practice and driving hospital consolidation. Restoring individual liberty requires repealing top-down federal insurance mandates and encouraging association health plans where small businesses can pool risk freely.",
        "conservative"
    ),
    (
        "State health departments report that enrollment across state insurance marketplaces grew moderately this quarter. Analysts note that federal tax credit extensions contributed to coverage stability, while rural hospital networks face ongoing staffing shortages and shifting reimbursement formulas.",
        "neutral"
    ),

    # --- CLIMATE & ENERGY (Liberal, Conservative, Neutral) ---
    (
        "The climate emergency represents an existential catastrophe that demands an immediate transition away from fossil fuels. Fossil fuel corporations have knowingly polluted our atmosphere and deceived the public for decades. We must enact a bold Green New Deal, mandate clean renewable energy standards, and heavily subsidize solar, wind, and public transit to protect frontline environmental justice communities.",
        "liberal"
    ),
    (
        "Unrealistic climate mandates and radical green regulations threaten American energy independence and inflict devastating energy inflation on working households. Nuclear power, domestic oil extraction, and natural gas pipelines must be expanded through regulatory relief to ensure national security and power the industrial economy.",
        "conservative"
    ),
    (
        "The International Energy Agency released its annual global energy review detailing trends in electricity generation. Renewables accounted for forty percent of newly added grid capacity, while natural gas remained the leading baseload power provider. The report highlights grid reliability challenges during peak seasonal transitions.",
        "neutral"
    ),
    (
        "Global warming is accelerating historic wildfires, droughts, and sea level rise, placing the burden squarely on marginalized populations. Federal subsidies for fossil fuel giants must end immediately, redirecting capital into community-owned clean microgrids and unionized green technology manufacturing jobs.",
        "liberal"
    ),
    (
        "Banning traditional energy drilling and strangling hydraulic fracturing with bureaucratic permits serves only to enrich hostile foreign adversaries while punishing domestic energy workers. American ingenuity in liquefied natural gas export facilities strengthens geopolitical alliances and provides reliable baseload power.",
        "conservative"
    ),
    (
        "Federal energy regulators held a public hearing on interstate transmission line permitting guidelines. Utility representatives and environmental advocates presented contrasting modeling estimates regarding infrastructure upgrade timelines, regional interconnection queues, and projected consumer electricity costs over the next decade.",
        "neutral"
    ),

    # --- TAXATION & ECONOMY (Liberal, Conservative, Neutral) ---
    (
        "Decades of supply-side trickle-down economics have exacerbated extreme wealth inequality, leaving billionaires with unprecedented riches while real wages for the working class stagnate. We must enact a progressive wealth tax, raise corporate income taxes, close offshore loopholes, and revitalize collective bargaining rights for labor unions.",
        "liberal"
    ),
    (
        "Economic growth flourishes when government lowers taxes, slashes burdensome regulations, and respects private property rights. Raising corporate tax rates penalizes domestic capital investment, discourages entrepreneurship, and forces corporations to relocate high-paying manufacturing jobs overseas.",
        "conservative"
    ),
    (
        "The Bureau of Labor Statistics released its monthly jobs report showing nonfarm payroll employment increased by two hundred thousand jobs, while the unemployment rate remained unchanged at four percent. Economic analysts observed moderate wage growth consistent with consumer price index benchmarks.",
        "neutral"
    ),
    (
        "Gig economy workers and warehouse laborers deserve dignity, living wages, and strong union protection against automated exploitation. Corporate executives receiving lavish stock buybacks must be held accountable through strict labor laws, paid family leave mandates, and fair compensation rules.",
        "liberal"
    ),
    (
        "Runaway government spending and runaway deficit financing are the primary drivers of devastating inflation. Fiscal discipline, spending caps, and pro-growth tax cuts are the proven path to unleash private sector productivity and protect the purchasing power of American families.",
        "conservative"
    ),
    (
        "The Federal Reserve concluded its policy meeting by leaving the benchmark federal funds rate steady. Committee members cited balanced signals between cooling inflation indicators and resilient consumer spending figures, projecting two future rate adjustments subject to incoming macroeconomic data.",
        "neutral"
    ),

    # --- IMMIGRATION & BORDER SECURITY (Liberal, Conservative, Neutral) ---
    (
        "Our immigration system requires compassionate, humane reform that provides an earned pathway to citizenship for millions of undocumented immigrants who contribute to our communities and economy. Cruel border militarization, family separations, and private detention facilities run counter to our core moral values as a welcoming nation of immigrants.",
        "liberal"
    ),
    (
        "A sovereign nation without secure borders ceases to be a nation. An open border policy invites dangerous cartels, illicit narcotics trafficking, and unchecked security risks. We must enforce federal immigration laws, finish the border barrier wall, and immediately deport those who violate national immigration sovereignty.",
        "conservative"
    ),
    (
        "Customs and Border Protection published quarterly border encounter statistics showing a six percent decrease in southwest border migrant encounters compared to the prior quarter. Government agencies and non-profit shelters continue coordinating logistical resources across transit hubs.",
        "neutral"
    ),
    (
        "Asylum seekers fleeing political violence and systemic poverty have a legally protected right to seek international refuge. Denying due process through arbitrary expedited removals and harsh enforcement policies violates both domestic asylum statutes and international humanitarian treaties.",
        "liberal"
    ),
    (
        "Rewarding illegal immigration through sanctuary city policies and amnesty proposals undermines the rule of law and drains public municipal resources. Legal immigrants who respect our immigration vetting process are treated unfairly when illegal entry is normalized and incentivized.",
        "conservative"
    ),
    (
        "Bipartisan legislative negotiators continue discussions over asylum adjudication funding and immigration court staffing levels. The pending proposal pairs increased border security surveillance technology with additional immigration judges to address the nationwide backlog of immigration cases.",
        "neutral"
    ),

    # --- JUDICIARY & CONSTITUTION (Liberal, Conservative, Neutral) ---
    (
        "An activist conservative majority on the Supreme Court has dismantled fundamental constitutional freedoms, rolling back voting rights protections, reproductive autonomy, and vital federal administrative authority to regulate corporate polluters. Congress must enact judicial ethics reforms and structural checks to restore democratic legitimacy.",
        "liberal"
    ),
    (
        "The Constitution must be interpreted according to its original public meaning and text rather than adapted by unelected progressive judges seeking to enact social policy from the bench. Originalist jurisprudence protects religious liberty, property rights, and the separation of powers against federal overreach.",
        "conservative"
    ),
    (
        "The Supreme Court concluded its term with eighty oral argument rulings spanning administrative law, intellectual property, and federal jurisdiction disputes. Legal scholars noted several narrow rulings where institutionalist votes crossed ideological blocs to reach statutory consensus.",
        "neutral"
    ),
    (
        "Voter suppression laws, aggressive partisan gerrymandering, and dark money campaign expenditures corrode our democratic republic. We need robust federal legislation like the Freedom to Vote Act to ensure every citizen has equal, unhindered access to the ballot box.",
        "liberal"
    ),
    (
        "State legislatures possess the constitutional prerogative under Article One to establish election integrity measures, including photo voter identification and clean voter rolls. Federalizing state elections strips local autonomy and undermines voter confidence in secure election outcomes.",
        "conservative"
    ),
    (
        "The bipartisan election commission released an audit reviewing absentee ballot verification protocols across eight battleground counties. The audit reported uniform adherence to state statutory deadlines and recommended updated scanning hardware for future election cycles.",
        "neutral"
    ),

    # --- EDUCATION & STUDENT LOANS (Liberal, Conservative, Neutral) ---
    (
        "Higher education is a public good, not a predatory financial scheme. Canceling crushing federal student loan debt and guaranteeing tuition-free public college will liberate a generation of young graduates from debt bondage and spark widespread social and economic mobility.",
        "liberal"
    ),
    (
        "Blanket student loan forgiveness transfers billions of dollars in debt from privileged college graduates onto hard-working blue-collar taxpayers who never attended university. School choice vouchers and educational freedom empower parents to rescue their children from failing public school monopolies.",
        "conservative"
    ),
    (
        "The Department of Education published data tracking post-secondary enrollment trends and federal loan repayment rates. Community college registrations rose by three percent nationwide, while default rates on direct federal loans fell following revised income-driven repayment guidelines.",
        "neutral"
    ),
    (
        "Public schools represent the bedrock of community life and must be fully funded with equitable resources for low-income school districts, competitive teacher salaries, and comprehensive mental health services, rather than defunded through privatization voucher schemes.",
        "liberal"
    ),
    (
        "Parents have an inalienable right to direct the moral and academic education of their children without government school bureaucracies pushing partisan ideological curricula. Educational tax credits create healthy competition that drives academic achievement across all institutions.",
        "conservative"
    ),
    (
        "State educational boards conducted statewide standardized assessments measuring mathematics and reading proficiency. Results showed steady recovery across suburban districts, while rural and urban districts displayed varied gaps influenced by instructional time and funding variations.",
        "neutral"
    ),

    # --- GUN POLICY & PUBLIC SAFETY (Liberal, Conservative, Neutral) ---
    (
        "Gun violence is an epidemic that tears apart American neighborhoods every day. Implementing common-sense gun safety reforms, including universal background checks, safe storage laws, and bans on military-style assault weapons and high-capacity magazines, will save innocent lives.",
        "liberal"
    ),
    (
        "The Second Amendment guarantees the individual right of law-abiding citizens to keep and bear arms for self-defense and security against tyranny. Disarming peaceful citizens through unconstitutional gun control laws leaves families defenseless against armed criminals who disregard all laws.",
        "conservative"
    ),
    (
        "The National Institute of Justice published its multi-year statistical review analyzing homicide trends across thirty major metropolitan areas. Violent crime rates decreased by eight percent overall, with variations attributed to community policing programs and targeted intervention strategies.",
        "neutral"
    ),
    (
        "Community violence intervention programs, sensible firearm licensing, and closing gun show loopholes are proven public safety measures supported by an overwhelming majority of citizens. We must hold gun manufacturers accountable for irresponsible marketing practices.",
        "liberal"
    ),
    (
        "Deterrence and robust law enforcement funding are the proven solutions to crime, not the erosion of constitutional liberties. Concealed carry permits allow responsible gun owners to protect themselves and deter active criminal threats in public spaces.",
        "conservative"
    ),
    (
        "Federal appellate judges heard arguments in a lawsuit challenging state firearm permit licensing procedures under recent Supreme Court precedent. Legal analysts expect the circuit court decision to center on historical analogues from early American statutes.",
        "neutral"
    ),

    # --- SOCIAL SAFETY NET & HOUSING (Liberal, Conservative, Neutral) ---
    (
        "The affordable housing crisis requires massive public investment in social housing, strict rent stabilization measures, and an end to corporate real estate investors buying up single-family starter homes. Housing is a human right, and municipal zoning must prioritize affordable community trusts.",
        "liberal"
    ),
    (
        "Excessive municipal regulations, rent control caps, and restrictive environmental zoning strangle private housing development, driving construction costs to unaffordable heights. Deregulating construction permits and incentivizing private builders will expand housing supply and restore affordability.",
        "conservative"
    ),
    (
        "The Department of Housing and Urban Development released its quarterly market overview tracking national mortgage rates, building permits, and median existing-home sales prices. Housing starts declined by two percent amid elevated financing rates, while rental vacancies remained steady.",
        "neutral"
    ),
    (
        "Expanding the Child Tax Credit lifted millions of American children out of poverty and demonstrated the transformative power of a strong social safety net. We must permanently expand family assistance, paid medical leave, and universal childcare subsidies.",
        "liberal"
    ),
    (
        "Welfare programs must include strict work requirements to encourage self-reliance, dignity, and workforce participation. Expanding unconditional entitlement spending creates generational government dependency and strains the national budget.",
        "conservative"
    ),
    (
        "A joint congressional fiscal committee evaluated the macroeconomic effects of temporary family tax credit expirations. The report balanced poverty reduction metrics against long-term federal deficit projections without endorsing specific legislative changes.",
        "neutral"
    ),

    # --- FOREIGN POLICY & DEFENSE (Liberal, Conservative, Neutral) ---
    (
        "American foreign policy must center human rights, diplomacy, and multilateral institutions over endless military spending and imperial interventions abroad. Ballooning Pentagon budgets divert hundreds of billions of dollars away from critical domestic investments in schools, healthcare, and infrastructure.",
        "liberal"
    ),
    (
        "Peace is secured through unmistakable military strength, deterrence, and an unwavering commitment to defending American national interests. Modernizing our defense capabilities, strengthening strategic alliances against authoritarian competitors like China and Russia, and funding missile defense are vital for global stability.",
        "conservative"
    ),
    (
        "Diplomatic delegations from forty nations concluded their annual security cooperation summit in Brussels. Member states issued a joint declaration outlining synchronized cybersecurity protocols, coordinated maritime patrol routes, and shared defense procurement frameworks.",
        "neutral"
    ),
    (
        "Endless military conflicts and unilateral sanctions devastate civilian populations and destabilize entire regions. De-escalation, international humanitarian law, and investing in global climate diplomacy are the true pillars of sustainable international security.",
        "liberal"
    ),
    (
        "Projecting American strength abroad prevents foreign aggression and safeguards freedom of international navigation. Retreating from international leadership invites autocratic regimes to expand territorial ambitions and disrupt international commercial maritime routes.",
        "conservative"
    ),
    (
        "The Congressional Research Service released an updated briefing paper on maritime security agreements across the Indo-Pacific. The analysis summarizes bilateral defense agreements, joint naval exercises, and trade corridor traffic volume statistics.",
        "neutral"
    ),

    # --- TECH REGULATION & CIVIL LIBERTIES (Liberal, Conservative, Neutral) ---
    (
        "Monopolistic tech conglomerates wield dangerous unchecked power over public discourse, exploit personal user privacy for predatory surveillance capitalism, and crush competitive startup innovation. We must break up Big Tech monopolies and enforce strict national consumer privacy standards.",
        "liberal"
    ),
    (
        "Big Tech platforms have engaged in systemic bias and censorship against conservative voices, suppressing dissenting opinions on public policy issues. We must protect free speech online, reform Section 230 protections, and hold tech monopolies accountable for ideological discrimination.",
        "conservative"
    ),
    (
        "The Federal Trade Commission and Justice Department antitrust division published updated merger guidelines for digital platform acquisitions. Industry representatives and consumer advocacy organizations filed public comments analyzing potential effects on digital market concentration.",
        "neutral"
    ),
    (
        "Facial recognition algorithms and algorithmic bias in automated hiring systems perpetuate systemic racial discrimination. The federal government must establish strict civil rights regulations governing artificial intelligence deployment in criminal justice and employment.",
        "liberal"
    ),
    (
        "Heavy-handed government mandates on emerging artificial intelligence technologies will stifle American innovation and surrender global technological leadership to geopolitical adversaries like China. Free enterprise and market competition should drive technological progress.",
        "conservative"
    ),
    (
        "A consortium of university researchers published a comprehensive study measuring model accuracy and latency across commercial language processing systems. The benchmark report outlined empirical performance across standard evaluation datasets without commercial endorsements.",
        "neutral"
    ),

    # --- TRADE & LABOR (Liberal, Conservative, Neutral) ---
    (
        "Neoliberal corporate trade agreements have hollowed out our industrial heartland, shipping millions of good-paying manufacturing jobs to low-wage countries with weak labor and environmental protections. Future trade policies must mandate strict international labor rights and carbon border adjustments.",
        "liberal"
    ),
    (
        "Free and fair trade expands consumer choice, lowers everyday household prices, and opens lucrative export markets for American agricultural producers and manufacturers. Imposing blanket tariffs acts as an inefficient tax on domestic consumers and triggers retaliatory trade disputes.",
        "conservative"
    ),
    (
        "The Department of Commerce reported international trade balance figures for the previous quarter. The trade deficit narrowed by three percent as agricultural exports increased, while imports of consumer electronics remained stable.",
        "neutral"
    ),
    (
        "Protecting workers' right to strike, eliminating right-to-work laws that weaken labor unions, and guaranteeing fair wages are essential to rebuilding a vibrant, egalitarian middle class.",
        "liberal"
    ),
    (
        "Right-to-work laws protect individual worker freedom by prohibiting compulsory union dues as a condition of employment. Flexible labor markets encourage business investment and job growth across dynamic state economies.",
        "conservative"
    ),
    (
        "The National Labor Relations Board released annual statistics summarizing union representation election filings and unfair labor practice charges. The report showed a four percent rise in representation petitions, primarily within logistics and service industries.",
        "neutral"
    ),

    # --- CRIMINAL JUSTICE & POLICING (Liberal, Conservative, Neutral) ---
    (
        "Systemic racism pervades our criminal justice system, driving mass incarceration of marginalized minorities for nonviolent offenses. True public safety requires ending cash bail, decriminalizing cannabis, dismantling the school-to-prison pipeline, and reallocating resources into mental health response teams.",
        "liberal"
    ),
    (
        "Soft-on-crime policies, progressive prosecutors refusing to enforce statutory penalties, and dismantling cash bail have led to spikes in retail theft and violent crime in metropolitan centers. We must back the police, fund law enforcement agencies, and ensure habitual criminals face mandatory sentences.",
        "conservative"
    ),
    (
        "State correctional departments released annual recidivism figures tracking individuals released from state custody over a three-year period. Recidivism rates varied across jurisdictions, with vocational training programs showing a five percent reduction in re-arrest rates.",
        "neutral"
    ),
    (
        "Solitary confinement and draconian mandatory minimum sentences inflict cruel punishment without improving rehabilitation. Restorative justice programs, educational access inside correctional facilities, and compassionate re-entry support provide the path toward humane justice.",
        "liberal"
    ),
    (
        "Upholding the rule of law requires empowering law enforcement officers with the legal protections and equipment needed to maintain order in our communities. Undermining police morale leads to officer attrition and compromised public safety.",
        "conservative"
    ),
    (
        "The Bureau of Justice Statistics released its annual report compiling data on law enforcement personnel numbers and local municipal budget allocations. The report documented municipal staffing levels across different population tiers.",
        "neutral"
    ),

    # --- BIOTECH & PHARMACEUTICAL PRICING ---
    (
        "Allowing Medicare to directly negotiate prescription drug prices is a crucial step toward dismantling price-gouging by Big Pharma. Patients should never have to ration lifesaving medications while pharmaceutical executives collect multi-million dollar bonuses funded by taxpayer research.",
        "liberal"
    ),
    (
        "Imposing artificial government price controls on pharmaceuticals destroys the economic incentives needed for risky research and development. American biotechnology leadership depends on patent protections that attract venture capital to discover the next generation of life-saving cures.",
        "conservative"
    ),
    (
        "The Government Accountability Office released a comparative study evaluating global prescription drug pricing structures. The report noted retail price differentials of forty percent between domestic and international markets, citing varying patent terms and price ceiling policies.",
        "neutral"
    ),
    (
        "Public health must override intellectual property monopolies during health emergencies. Waiving vaccine and antiviral patent barriers enables developing nations to produce affordable generic medicines for vulnerable populations worldwide.",
        "liberal"
    ),
    (
        "Weakening intellectual property rights through compulsory licensing amounts to theft of intellectual property and disincentivizes American pharmaceutical innovation. Strong patent enforcement ensures developers can recoup substantial clinical trial costs.",
        "conservative"
    ),
    (
        "The Food and Drug Administration published its annual generic drug review showing five hundred new generic approvals over the fiscal year. Median review times decreased by two months following updated regulatory guidance on bioequivalence testing.",
        "neutral"
    ),

    # --- FINANCIAL REGULATION & BANKING ---
    (
        "Wall Street banks continue gambling with systemic risks, expecting public bailouts whenever speculative bubbles burst. We must reinstate modern Glass-Steagall protections, cap consumer credit card interest rates, and penalize predatory financial institutions that exploit ordinary depositors.",
        "liberal"
    ),
    (
        "Overreaching financial regulations like Dodd-Frank strangle regional and community banks under mountains of compliance red tape. Eliminating burdensome capital constraints and fostering free-market financial innovation will expand credit for small business job creators.",
        "conservative"
    ),
    (
        "The Federal Deposit Insurance Corporation issued its quarterly banking performance overview. Commercial banks reported a three percent increase in net interest income, alongside stable loan default ratios across commercial real estate portfolios.",
        "neutral"
    ),
    (
        "Unregulated cryptocurrency platforms operate like unregulated shadow banks and predatory financial casinos that leave retail investors vulnerable to rampant fraud. Regulators must enforce rigorous transparency, consumer disclosures, and capital reserve requirements.",
        "liberal"
    ),
    (
        "Decentralized finance and cryptocurrency represent a technological frontier that protects individual monetary freedom from central bank inflation and currency devaluation. Government overregulation will simply drive fintech pioneers to friendlier offshore jurisdictions.",
        "conservative"
    ),
    (
        "The Securities and Exchange Commission published an advisory staff bulletin summarizing existing registration obligations for digital asset intermediaries and secondary trading platforms under federal securities statutes.",
        "neutral"
    ),

    # --- INFRASTRUCTURE & TRANSPORTATION ---
    (
        "Decades of disinvestment have left our public mass transit systems deteriorating while suburban car culture accelerates carbon emissions. Massive federal investment in high-speed rail, electrified urban buses, and equitable public transit connects neglected communities to good jobs.",
        "liberal"
    ),
    (
        "Pouring billions of taxpayer dollars into high-speed rail boondoggles with cost overruns is fiscal insanity. Private sector toll roads, competitive airport infrastructure partnerships, and pipeline modernization deliver far higher economic returns without swelling the national debt.",
        "conservative"
    ),
    (
        "The Federal Highway Administration allocated forty billion dollars in formula grants across state transportation departments. State agency audits indicated that bridge repairs and highway resurfacing projects accounted for sixty percent of committed contracts.",
        "neutral"
    ),
    (
        "Highway expansions through low-income neighborhoods exacerbate air pollution and displace historically disadvantaged communities. Transportation policy must prioritize pedestrian safety, bike networks, and zero-emission public mobility.",
        "liberal"
    ),
    (
        "Streamlining environmental review under the National Environmental Policy Act is essential to build pipelines, bridges, and highways rapidly without endless obstructionist litigation by green activist groups.",
        "conservative"
    ),
    (
        "The Department of Transportation released its annual travel volume statistics, noting commercial air travel passengers surpassed pre-pandemic highs while highway freight mileage increased by one percent.",
        "neutral"
    ),

    # --- AGRICULTURE & WATER POLICY ---
    (
        "Factory farming mega-corporations monopolize agriculture, abuse animals, contaminate local waterways with toxic runoff, and squeeze independent family farmers into bankruptcy. We must enforce antitrust laws in meatpacking and support regenerative organic farming.",
        "liberal"
    ),
    (
        "Excessive EPA water regulations and onerous endangered species listings infringe on private property rights and disrupt American food production. Empowering farmers and ranchers with regulatory certainty and free-market water rights is vital for national food security.",
        "conservative"
    ),
    (
        "The Department of Agriculture published its farm income forecast projecting net farm income to reach one hundred forty billion dollars. The report documented decreases in commodity grain prices offset by lower fertilizer input expenditures.",
        "neutral"
    ),
    (
        "Agricultural workers, disproportionately immigrant laborers, endure extreme heat, dangerous pesticide exposure, and substandard wages without federal overtime protections. Enacting strict workplace heat safety standards and labor rights for farmworkers is an urgent moral necessity.",
        "liberal"
    ),
    (
        "Family farms face crippling death taxes that force generational heirs to liquidate acreage just to satisfy federal estate tax bills. Eliminating the estate tax and expanding agricultural trade exports will preserve the family farming tradition.",
        "conservative"
    ),
    (
        "The Bureau of Reclamation issued its seasonal reservoir allocation bulletin for the Colorado River basin. Water release volumes were adjusted according to elevation tiers and interstate drought contingency agreement schedules.",
        "neutral"
    ),
]


def generate_dataset(output_path: Path) -> pd.DataFrame:
    """Creates and saves the news bias CSV dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ARTICLES, columns=["text", "bias"])
    df["title"] = [f"Opinion Piece {i+1}" for i in range(len(df))]

    print(f"Generated dataset with {len(df)} records.")
    print("Class breakdown:\n", df["bias"].value_counts())

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved dataset to: {output_path}")
    return df


def generate_coherent_glove_mini(
    df: pd.DataFrame,
    output_path: Path,
    dim: int = 50,
    seed: int = 42
) -> None:
    """
    Generates a starter GloVe embedding text file where words have coherent,
    topic-oriented semantic representations in a 50-dimensional space.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1. Extract vocabulary from the corpus
    vocab = set()
    for text in df["text"]:
        words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
        for w in words:
            if len(w) >= 2:
                vocab.add(w)

    # Additional standard English and political vocabulary
    common_words = [
        "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was",
        "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
        "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
        "all", "were", "we", "when", "your", "can", "said", "there", "use", "an",
        "each", "which", "she", "do", "how", "their", "if", "will", "up", "other",
        "about", "out", "many", "then", "them", "these", "so", "some", "her", "would",
        "make", "like", "him", "into", "time", "has", "look", "two", "more", "write",
        "go", "see", "number", "no", "way", "could", "people", "my", "than", "first",
        "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down",
        "day", "did", "get", "come", "made", "may", "part", "liberal", "conservative", "neutral",
        "left", "right", "center",
        "bias", "opinion", "editorial", "article", "stance", "policy", "politics",
        "democrat", "republican", "independent", "bipartisan", "congress", "senate",
        "president", "judicial", "economy", "market", "liberty", "equality", "freedom"
    ]
    vocab.update(common_words)

    # 2. Semantic topic anchors (subspaces) to give words coherent geometric properties
    # Stance semantic axes:
    # Dimension 0-9: Liberal-leaning semantic cluster (equality, welfare, public, justice, regulation, climate)
    # Dimension 10-19: Conservative-leaning semantic cluster (liberty, market, private, sovereignty, border, constitution)
    # Dimension 20-29: Neutral reporting cluster (report, study, data, statistics, balance, officials)
    # Dimension 30-49: General lexical & syntactic embedding components
    liberal_keywords = {
        "progressive", "equality", "medicare", "worker", "unions", "climate", "emergency",
        "inequality", "billionaires", "taxing", "universal", "systemic", "justice", "green",
        "public", "reform", "predatory", "oppressed", "marginalized", "humane", "asylum",
        "affordable", "welfare", "corporations", "monopolies", "subsidies", "disproportionate",
        "reallocation", "restitution", "living", "wage", "decarbonization", "grassroots", "liberal",
        "solidarity", "redistribution", "fairness", "medicaid", "social", "collective"
    }

    conservative_keywords = {
        "liberty", "freedom", "market", "deregulation", "deregulate", "deregulating", "competition",
        "sovereignty", "originalism", "constitution", "constitutional", "border", "deterrence",
        "enforcement", "drilling", "tariffs", "vouchers", "taxpayers", "spending", "discipline",
        "individual", "amendment", "second", "self-defense", "police", "sovereign", "sanctuary",
        "bureaucratic", "socialism", "unconstitutional", "tyranny", "patriotism", "property",
        "conservative", "enterprise", "burdensome", "cutting", "pro-growth", "slashing", "capitalism"
    }

    neutral_keywords = {
        "report", "analysis", "statistics", "department", "bureau", "cbo", "data",
        "quarterly", "study", "annual", "committee", "officials", "economists",
        "moderate", "benchmark", "projections", "balance", "hearing", "audit",
        "consensus", "trends", "rates", "indices", "adjudication", "survey", "indicators", "neutral",
        "percent", "published", "released", "findings", "statistics", "methodology"
    }

    print(f"Generating coherent GloVe embeddings for {len(vocab)} words...")

    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            word_seed = abs(hash(word)) % (2**31)
            rng = np.random.default_rng(word_seed)
            vec = rng.normal(loc=0.0, scale=0.15, size=dim)

            # Boost stance-aligned dimensions for semantic coherence
            if word in liberal_keywords or any(k in word for k in ["union", "equal", "public", "green", "climat", "welfare"]):
                vec[0:10] += rng.uniform(0.8, 1.4, size=10)
                vec[10:20] -= rng.uniform(0.3, 0.6, size=10)
            elif word in conservative_keywords or any(k in word for k in ["libert", "market", "border", "sovereign", "enterpris", "constitut"]):
                vec[10:20] += rng.uniform(0.8, 1.4, size=10)
                vec[0:10] -= rng.uniform(0.3, 0.6, size=10)
            elif word in neutral_keywords or any(k in word for k in ["statist", "report", "percent", "audit", "project"]):
                vec[20:30] += rng.uniform(0.8, 1.4, size=10)

            # Normalize vector to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            line = f"{word} " + " ".join(f"{val:.6f}" for val in vec)
            f.write(line + "\n")

    print(f"Saved GloVe mini embeddings to: {output_path}")


def generate_sample_article(output_path: Path) -> None:
    """Generates a sample article for CLI file prediction tests."""
    sample_text = (
        "The proposed regulatory overhaul strikes at the heart of free enterprise and market competition. "
        "By imposing centralized federal mandates and heavy compliance costs on independent energy producers, "
        "the administration risks stifling domestic capital investment and weakening American energy independence. "
        "True economic growth requires deregulation, fiscal discipline, and unleashing the power of private innovation."
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"Saved sample article for CLI testing to: {output_path}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    emb_dir = data_dir / "embeddings"

    csv_path = data_dir / "news_bias_dataset.csv"
    glove_path = emb_dir / "glove_mini.50d.txt"
    sample_path = data_dir / "sample_article.txt"

    df = generate_dataset(csv_path)
    generate_coherent_glove_mini(df, glove_path, dim=50)
    generate_sample_article(sample_path)
