export type DocName = string

export type OutlineItem = {
  level: 'H1'|'H2'|'H3'|'H4'|'H5'|'H6'
  text: string
  page: number
}
export type OutlineResponse = {
  title: string
  outline: OutlineItem[]
}

export type RecommendationItem = {
  document: string
  section_title: string
  page_number: number
  score: number
  snippet: string
}
export type RecommendationsResponse = {
  items: RecommendationItem[]
  domain: string
  query_used?: string
}

export type InsightsResponse = {
  insight: string
  used_items: RecommendationItem[]
}
