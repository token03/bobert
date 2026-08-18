import { useMutation } from '@tanstack/react-query'
import { recommendBeatmaps } from '../../shared/api'
import type { RecommendRequest } from '../../shared/types'

export function useRecommend() {
  return useMutation({
    mutationFn: (request: RecommendRequest) => recommendBeatmaps(request),
  })
}
