// Package controllers stores all the controllers for the Gin router.
package controllers

import (
	"TaipeiCityDashboardBE/app/services/ai"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/tmc/langchaingo/llms"
)

// NavigationGeoJSONRequest represents the request body for navigation GeoJSON endpoint
type NavigationGeoJSONRequest struct {
	Type     string                 `json:"type" binding:"required,eq=FeatureCollection"`
	Features []interface{}          `json:"features" binding:"required"`
	Bbox     []float64              `json:"bbox,omitempty"`
	Crs      map[string]interface{} `json:"crs,omitempty"`
	Files    []string               `json:"files,omitempty"`
}

// NavigationGeoJSONResponse represents the response body for navigation GeoJSON endpoint
type NavigationGeoJSONResponse struct {
	Status        string                 `json:"status"`
	FeatureCount  int                    `json:"feature_count"`
	GeometryTypes map[string]int         `json:"geometry_types"`
	BoundingBox   []float64              `json:"bounding_box,omitempty"`
	Properties    map[string]interface{} `json:"properties,omitempty"`
	Overlaps      []OverlapResult        `json:"overlaps,omitempty"`
	Message       string                 `json:"message,omitempty"`
}

// OverlapResult holds reference features that the route crosses
type OverlapResult struct {
	ReferenceFile       string        `json:"reference_file"`
	OverlappingFeatures []interface{} `json:"overlapping_features"` // Reference features crossed by the route
	OverlapCount        int           `json:"overlap_count"`
}

// Cache for loaded GeoJSON files to avoid repeated disk reads
var geoJSONCache = map[string]interface{}{}
var compiledGeoJSONCache = map[string]*compiledReferenceFile{}
var cacheMutex sync.RWMutex

type geoBBox struct {
	minX  float64
	minY  float64
	maxX  float64
	maxY  float64
	valid bool
}

type compiledSegment struct {
	a    [2]float64
	b    [2]float64
	bbox geoBBox
}

type compiledPolygonRing struct {
	points   [][2]float64
	segments []compiledSegment
	bbox     geoBBox
}

type compiledReferenceFeature struct {
	raw          interface{}
	bbox         geoBBox
	polygonRings []compiledPolygonRing
	lineSegments []compiledSegment
	points       [][2]float64
}

type compiledReferenceFile struct {
	features []compiledReferenceFeature
}

type compiledRouteFeature struct {
	points   [][2]float64
	segments []compiledSegment
	bbox     geoBBox
	cellSize float64
	grid     map[[2]int][]int
}

// HandleNavigationGeoJSON processes incoming navigation GeoJSON data
// POST /api/v1/navigation/geojson
func HandleNavigationGeoJSON(c *gin.Context) {
	var navReq NavigationGeoJSONRequest
	if err := c.ShouldBindJSON(&navReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Invalid GeoJSON format: " + err.Error(),
		})
		return
	}

	response := NavigationGeoJSONResponse{
		Status:        "success",
		FeatureCount:  len(navReq.Features),
		GeometryTypes: make(map[string]int),
		Properties:    make(map[string]interface{}),
		Message:       "GeoJSON processed successfully",
	}

	var allCoords [][]float64
	hasValidCoords := false

	for _, feature := range navReq.Features {
		featureMap, ok := feature.(map[string]interface{})
		if !ok {
			continue
		}

		geometry, ok := featureMap["geometry"].(map[string]interface{})
		if !ok {
			continue
		}

		geomType, ok := geometry["type"].(string)
		if !ok {
			continue
		}

		response.GeometryTypes[geomType]++

		if coordinates, ok := geometry["coordinates"].(interface{}); ok {
			coords := extractCoordinates(coordinates)
			if len(coords) > 0 {
				allCoords = append(allCoords, coords...)
				hasValidCoords = true
			}
		}

		if properties, ok := featureMap["properties"].(map[string]interface{}); ok {
			for key, value := range properties {
				if _, exists := response.Properties[key]; !exists {
					response.Properties[key] = value
				}
			}
		}
	}

	if hasValidCoords && len(allCoords) > 0 {
		minX, minY, maxX, maxY := allCoords[0][0], allCoords[0][1], allCoords[0][0], allCoords[0][1]
		for _, coord := range allCoords {
			if len(coord) >= 2 {
				if coord[0] < minX {
					minX = coord[0]
				}
				if coord[0] > maxX {
					maxX = coord[0]
				}
				if coord[1] < minY {
					minY = coord[1]
				}
				if coord[1] > maxY {
					maxY = coord[1]
				}
			}
		}
		response.BoundingBox = []float64{minX, minY, maxX, maxY}
	}

	if len(navReq.Files) > 0 {
		overlapResults, err := processOverlapDetection(navReq.Features, navReq.Files)
		if err != nil {
			response.Overlaps = []OverlapResult{}
		} else {
			response.Overlaps = overlapResults
		}
	} else {
		files, err := getGeoJSONFilesInPipeline()
		if err != nil {
			response.Overlaps = []OverlapResult{}
		} else if len(files) > 0 {
			overlapResults, err := processOverlapDetection(navReq.Features, files)
			if err != nil {
				response.Overlaps = []OverlapResult{}
			} else {
				response.Overlaps = overlapResults
			}
		}
	}

	// ── AI Analysis ───────────────────────────────────────────────────────────
	// https://docs.twcloud.ai/docs/user-guides/twcc/afs/api-and-parameters/api-parameter-information#模型說明

	// 1. Collect only the properties of each overlapping feature across all files
	var overlapProps []map[string]interface{}
	for _, overlap := range response.Overlaps {
		for _, feat := range overlap.OverlappingFeatures {
			featMap, ok := feat.(map[string]interface{})
			if !ok {
				continue
			}
			props, ok := featMap["properties"].(map[string]interface{})
			if !ok {
				continue
			}
			overlapProps = append(overlapProps, props)
		}
	}

	for i, j := 0, len(overlapProps)-1; i < j; i, j = i+1, j-1 {
		overlapProps[i], overlapProps[j] = overlapProps[j], overlapProps[i]
	}

	// 2. Build prompt from overlap properties
	propsJSON, err := json.Marshal(overlapProps)
	if err != nil {
		log.Printf("[Navigation] 無法序列化 overlap properties: %v", err)
		propsJSON = []byte("[]")
	}
	aiPrompt := fmt.Sprintf("以下是路線經過地區的屬性資料，從起點至終點按順序簡述哪些路段會下雨，並提醒使用者小心駕駛，全文不超過150字：\n%s", string(propsJSON))

	// 3. Session ID — prefer X-Request-ID header; fall back to a new random ID
	sessionID := c.GetHeader("X-Request-ID")
	if sessionID == "" {
		sessionID = fmt.Sprintf("nav_%s", generateSimpleID())
	}

	// 4. Fire the AI request
	req := ai.AIChatRequest{
		SessionID: sessionID,
		IPAddress: c.ClientIP(),
		Messages: []llms.MessageContent{
			{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: aiPrompt},
				},
			},
		},
	}

	logEntry, aiErr := ai.ChatWithTWCC(c.Request.Context(), req)
	if aiErr != nil {
		// AI failure is non-fatal — return overlap results without AI analysis
		log.Printf("[Navigation] AI 分析失敗: %v", aiErr)
		c.JSON(http.StatusOK, response)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  response.Status,
		"message": response.Message,
		"ai": gin.H{
			"session": logEntry.SessionID,
			"content": logEntry.Answer,
			"usage": gin.H{
				"input_tokens":  logEntry.InputTokens,
				"output_tokens": logEntry.OutputTokens,
				"total_tokens":  logEntry.TotalTokens,
			},
			"tool_used":  logEntry.ToolUsed,
			"latency_ms": logEntry.LatencyMS,
			"model":      logEntry.Model,
			"provider":   logEntry.Provider,
		},
	})
}

// generateSimpleID returns a short pseudo-random hex string for session IDs.
func generateSimpleID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", os.Getpid())
	}
	return fmt.Sprintf("%x", b)
}

// processOverlapDetection finds reference features that the incoming route crosses.
// For each reference file, it returns the features (e.g. districts, zones) that
// the route geometry intersects.
func processOverlapDetection(navFeatures []interface{}, referenceFiles []string) ([]OverlapResult, error) {
	routes := compileNavigationRouteFeatures(navFeatures)
	if len(routes) == 0 || len(referenceFiles) == 0 {
		return []OverlapResult{}, nil
	}

	resultSlots := make([]OverlapResult, len(referenceFiles))
	hasResult := make([]bool, len(referenceFiles))
	var wg sync.WaitGroup

	for index, refFile := range referenceFiles {
		wg.Add(1)
		go func(index int, refFile string) {
			defer wg.Done()

			compiledFile, err := loadCompiledGeoJSONFile(refFile)
			if err != nil {
				log.Printf("[Navigation] 無法載入參考檔案 %s: %v", refFile, err)
				return
			}

			crossedFeatures := findCrossedReferenceFeatures(routes, compiledFile.features)
			if len(crossedFeatures) == 0 {
				return
			}

			resultSlots[index] = OverlapResult{
				ReferenceFile:       refFile,
				OverlappingFeatures: crossedFeatures,
				OverlapCount:        len(crossedFeatures),
			}
			hasResult[index] = true
		}(index, refFile)
	}

	wg.Wait()

	results := make([]OverlapResult, 0, len(referenceFiles))
	for index, result := range resultSlots {
		if hasResult[index] {
			results = append(results, result)
		}
	}

	return results, nil
}

func compileNavigationRouteFeatures(navFeatures []interface{}) []compiledRouteFeature {
	var routes []compiledRouteFeature

	for _, navFeature := range navFeatures {
		routeGeom := extractGeometry(navFeature)
		if routeGeom == nil {
			continue
		}

		routeType, _ := routeGeom["type"].(string)
		switch routeType {
		case "LineString":
			if route := compileRouteLine(extractPointList(routeGeom["coordinates"])); route != nil {
				routes = append(routes, *route)
			}
		case "MultiLineString":
			for _, line := range extractLineStrings(routeGeom["coordinates"]) {
				if route := compileRouteLine(line); route != nil {
					routes = append(routes, *route)
				}
			}
		}
	}

	return routes
}

func compileRouteLine(points [][2]float64) *compiledRouteFeature {
	if len(points) < 2 {
		return nil
	}

	segments := compileSegments(points, false)
	if len(segments) == 0 {
		return nil
	}

	route := &compiledRouteFeature{
		points:   points,
		segments: segments,
		bbox:     bboxFromPoints(points),
		grid:     make(map[[2]int][]int, len(segments)*2),
	}
	route.cellSize = routeGridCellSize(route.bbox)
	route.indexSegments()
	return route
}

func loadCompiledGeoJSONFile(filename string) (*compiledReferenceFile, error) {
	cacheMutex.RLock()
	if compiled, exists := compiledGeoJSONCache[filename]; exists {
		cacheMutex.RUnlock()
		return compiled, nil
	}
	cacheMutex.RUnlock()

	refData, err := loadGeoJSONFile(filename)
	if err != nil {
		return nil, err
	}

	refFeatures, err := extractFeaturesFromGeoJSON(refData)
	if err != nil {
		return nil, fmt.Errorf("無法解析 features %s: %w", filename, err)
	}

	compiled := &compiledReferenceFile{
		features: compileReferenceFeatures(refFeatures),
	}

	cacheMutex.Lock()
	compiledGeoJSONCache[filename] = compiled
	cacheMutex.Unlock()

	return compiled, nil
}

func compileReferenceFeatures(refFeatures []interface{}) []compiledReferenceFeature {
	compiledFeatures := make([]compiledReferenceFeature, 0, len(refFeatures))

	for _, refFeature := range refFeatures {
		refGeom := extractGeometry(refFeature)
		if refGeom == nil {
			continue
		}

		refType, _ := refGeom["type"].(string)
		compiled := compiledReferenceFeature{
			raw:  refFeature,
			bbox: bboxFromCoordinates(refGeom["coordinates"]),
		}
		if !compiled.bbox.valid {
			continue
		}

		switch refType {
		case "Polygon":
			compiled.polygonRings = compilePolygonRings(refGeom["coordinates"])
		case "MultiPolygon":
			compiled.polygonRings = compileMultiPolygonRings(refGeom["coordinates"])
		case "LineString":
			compiled.lineSegments = compileSegments(extractPointList(refGeom["coordinates"]), false)
		case "MultiLineString":
			for _, line := range extractLineStrings(refGeom["coordinates"]) {
				compiled.lineSegments = append(
					compiled.lineSegments,
					compileSegments(line, false)...,
				)
			}
		case "Point":
			compiled.points = append(compiled.points, extractSinglePoint(refGeom["coordinates"]))
		case "MultiPoint":
			compiled.points = extractPointList(refGeom["coordinates"])
		}

		if len(compiled.polygonRings) == 0 &&
			len(compiled.lineSegments) == 0 &&
			len(compiled.points) == 0 {
			continue
		}
		compiledFeatures = append(compiledFeatures, compiled)
	}

	return compiledFeatures
}

func compilePolygonRings(coords interface{}) []compiledPolygonRing {
	rings := extractRings(coords)
	if len(rings) == 0 {
		return nil
	}
	return compileOuterPolygonRing(rings[0])
}

func compileMultiPolygonRings(coords interface{}) []compiledPolygonRing {
	polygons := extractMultiPolygonRings(coords)
	compiledRings := make([]compiledPolygonRing, 0, len(polygons))
	for _, rings := range polygons {
		if len(rings) == 0 {
			continue
		}
		compiledRings = append(compiledRings, compileOuterPolygonRing(rings[0])...)
	}
	return compiledRings
}

func compileOuterPolygonRing(points [][2]float64) []compiledPolygonRing {
	if len(points) < 3 {
		return nil
	}

	segments := compileSegments(points, true)
	if len(segments) == 0 {
		return nil
	}

	return []compiledPolygonRing{
		{
			points:   points,
			segments: segments,
			bbox:     bboxFromPoints(points),
		},
	}
}

func compileSegments(points [][2]float64, closeRing bool) []compiledSegment {
	if len(points) < 2 {
		return nil
	}

	segmentCount := len(points) - 1
	if closeRing && points[0] != points[len(points)-1] {
		segmentCount++
	}

	segments := make([]compiledSegment, 0, segmentCount)
	for i := 0; i < len(points)-1; i++ {
		segments = append(segments, newCompiledSegment(points[i], points[i+1]))
	}
	if closeRing && points[0] != points[len(points)-1] {
		segments = append(segments, newCompiledSegment(points[len(points)-1], points[0]))
	}
	return segments
}

func newCompiledSegment(a, b [2]float64) compiledSegment {
	return compiledSegment{
		a:    a,
		b:    b,
		bbox: bboxFromPoints([][2]float64{a, b}),
	}
}

func findCrossedReferenceFeatures(routes []compiledRouteFeature, refFeatures []compiledReferenceFeature) []interface{} {
	crossedFeatures := make([]interface{}, 0)

	for _, refFeature := range refFeatures {
		for routeIndex := range routes {
			if routeIntersectsCompiledFeature(&routes[routeIndex], &refFeature) {
				crossedFeatures = append(crossedFeatures, refFeature.raw)
				break
			}
		}
	}

	return crossedFeatures
}

func routeIntersectsCompiledFeature(route *compiledRouteFeature, refFeature *compiledReferenceFeature) bool {
	if route == nil || refFeature == nil || !geoBoxesOverlap(route.bbox, refFeature.bbox) {
		return false
	}

	for _, refPoint := range refFeature.points {
		if routeIntersectsPoint(route, refPoint) {
			return true
		}
	}

	for _, refSegment := range refFeature.lineSegments {
		if routeIntersectsSegment(route, refSegment) {
			return true
		}
	}

	for _, ring := range refFeature.polygonRings {
		if routeIntersectsPolygonRing(route, ring) {
			return true
		}
	}

	return false
}

func routeIntersectsPolygonRing(route *compiledRouteFeature, ring compiledPolygonRing) bool {
	if !geoBoxesOverlap(route.bbox, ring.bbox) {
		return false
	}

	for _, edge := range ring.segments {
		if routeIntersectsSegment(route, edge) {
			return true
		}
	}

	for _, routePoint := range route.points {
		if bboxContainsPoint(ring.bbox, routePoint) && pointInPolygon(routePoint, ring.points) {
			return true
		}
	}

	return false
}

func routeIntersectsSegment(route *compiledRouteFeature, refSegment compiledSegment) bool {
	if !geoBoxesOverlap(route.bbox, refSegment.bbox) {
		return false
	}

	for _, routeSegment := range route.querySegments(refSegment.bbox) {
		if geoBoxesOverlap(routeSegment.bbox, refSegment.bbox) &&
			segmentsIntersect(routeSegment.a, routeSegment.b, refSegment.a, refSegment.b) {
			return true
		}
	}
	return false
}

func routeIntersectsPoint(route *compiledRouteFeature, refPoint [2]float64) bool {
	const threshold = 0.0001
	pointBox := expandBBox(bboxFromPoints([][2]float64{refPoint}), threshold)
	if !geoBoxesOverlap(route.bbox, pointBox) {
		return false
	}

	for _, routeSegment := range route.querySegments(pointBox) {
		if pointNearSegment(refPoint, routeSegment.a, routeSegment.b, threshold) {
			return true
		}
	}
	return false
}

func routeGridCellSize(bbox geoBBox) float64 {
	const minCellSize = 0.001
	const maxCellSize = 0.01

	width := bbox.maxX - bbox.minX
	height := bbox.maxY - bbox.minY
	size := math.Max(width, height) / 80
	if !bbox.valid || !isFinitePositive(size) {
		return minCellSize
	}
	return math.Max(minCellSize, math.Min(maxCellSize, size))
}

func isFinitePositive(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

func (route *compiledRouteFeature) indexSegments() {
	if route == nil || !route.bbox.valid || route.cellSize <= 0 {
		return
	}

	for index, segment := range route.segments {
		minCellX, minCellY, maxCellX, maxCellY := route.cellRange(segment.bbox)
		for cellX := minCellX; cellX <= maxCellX; cellX++ {
			for cellY := minCellY; cellY <= maxCellY; cellY++ {
				key := [2]int{cellX, cellY}
				route.grid[key] = append(route.grid[key], index)
			}
		}
	}
}

func (route *compiledRouteFeature) querySegments(bbox geoBBox) []compiledSegment {
	if route == nil || !bbox.valid || !geoBoxesOverlap(route.bbox, bbox) {
		return nil
	}

	minCellX, minCellY, maxCellX, maxCellY := route.cellRange(bbox)
	seen := make(map[int]struct{})
	segments := make([]compiledSegment, 0)

	for cellX := minCellX; cellX <= maxCellX; cellX++ {
		for cellY := minCellY; cellY <= maxCellY; cellY++ {
			for _, segmentIndex := range route.grid[[2]int{cellX, cellY}] {
				if _, exists := seen[segmentIndex]; exists {
					continue
				}
				seen[segmentIndex] = struct{}{}
				segments = append(segments, route.segments[segmentIndex])
			}
		}
	}

	return segments
}

func (route *compiledRouteFeature) cellRange(bbox geoBBox) (int, int, int, int) {
	minCellX := int(math.Floor((bbox.minX - route.bbox.minX) / route.cellSize))
	minCellY := int(math.Floor((bbox.minY - route.bbox.minY) / route.cellSize))
	maxCellX := int(math.Floor((bbox.maxX - route.bbox.minX) / route.cellSize))
	maxCellY := int(math.Floor((bbox.maxY - route.bbox.minY) / route.cellSize))
	return minCellX, minCellY, maxCellX, maxCellY
}

func bboxFromCoordinates(coords interface{}) geoBBox {
	var bbox geoBBox

	var visit func(interface{})
	visit = func(value interface{}) {
		switch typedValue := value.(type) {
		case []interface{}:
			if len(typedValue) >= 2 {
				x, okX := typedValue[0].(float64)
				y, okY := typedValue[1].(float64)
				if okX && okY {
					bbox = extendBBox(bbox, [2]float64{x, y})
					return
				}
			}
			for _, item := range typedValue {
				visit(item)
			}
		case map[string]interface{}:
			for _, item := range typedValue {
				visit(item)
			}
		}
	}

	visit(coords)
	return bbox
}

func bboxFromPoints(points [][2]float64) geoBBox {
	var bbox geoBBox
	for _, point := range points {
		bbox = extendBBox(bbox, point)
	}
	return bbox
}

func extendBBox(bbox geoBBox, point [2]float64) geoBBox {
	if !bbox.valid {
		return geoBBox{
			minX:  point[0],
			minY:  point[1],
			maxX:  point[0],
			maxY:  point[1],
			valid: true,
		}
	}
	if point[0] < bbox.minX {
		bbox.minX = point[0]
	}
	if point[0] > bbox.maxX {
		bbox.maxX = point[0]
	}
	if point[1] < bbox.minY {
		bbox.minY = point[1]
	}
	if point[1] > bbox.maxY {
		bbox.maxY = point[1]
	}
	return bbox
}

func expandBBox(bbox geoBBox, amount float64) geoBBox {
	if !bbox.valid {
		return bbox
	}
	bbox.minX -= amount
	bbox.minY -= amount
	bbox.maxX += amount
	bbox.maxY += amount
	return bbox
}

func geoBoxesOverlap(a, b geoBBox) bool {
	if !a.valid || !b.valid {
		return false
	}
	if a.maxX < b.minX || b.maxX < a.minX {
		return false
	}
	if a.maxY < b.minY || b.maxY < a.minY {
		return false
	}
	return true
}

func bboxContainsPoint(bbox geoBBox, point [2]float64) bool {
	return bbox.valid &&
		bbox.minX <= point[0] && point[0] <= bbox.maxX &&
		bbox.minY <= point[1] && point[1] <= bbox.maxY
}

// ─── Geometry Intersection ────────────────────────────────────────────────────

// routeIntersectsFeature checks whether a route feature (LineString) intersects
// a reference feature (Polygon, MultiPolygon, LineString, or Point).
func routeIntersectsFeature(routeFeature, refFeature interface{}) bool {
	routeGeom := extractGeometry(routeFeature)
	refGeom := extractGeometry(refFeature)
	if routeGeom == nil || refGeom == nil {
		return false
	}

	// Quick bounding box rejection
	if !boundingBoxesOverlap(routeGeom["coordinates"], refGeom["coordinates"]) {
		return false
	}

	routeType, _ := routeGeom["type"].(string)
	refType, _ := refGeom["type"].(string)

	// Build route segments from LineString
	var routeSegments [][2][2]float64
	if routeType == "LineString" {
		pts := extractPointList(routeGeom["coordinates"])
		for i := 0; i < len(pts)-1; i++ {
			routeSegments = append(routeSegments, [2][2]float64{pts[i], pts[i+1]})
		}
	}

	switch refType {
	case "Polygon":
		rings := extractRings(refGeom["coordinates"])
		if len(rings) == 0 {
			return false
		}
		outerRing := rings[0]
		// Check segment vs polygon edge intersection
		for _, seg := range routeSegments {
			for i := 0; i < len(outerRing)-1; i++ {
				if segmentsIntersect(seg[0], seg[1], outerRing[i], outerRing[i+1]) {
					return true
				}
			}
		}
		// Check if any route point is inside the polygon
		pts := extractPointList(routeGeom["coordinates"])
		for _, pt := range pts {
			if pointInPolygon(pt, outerRing) {
				return true
			}
		}

	case "MultiPolygon":
		polygons := extractMultiPolygonRings(refGeom["coordinates"])
		for _, rings := range polygons {
			if len(rings) == 0 {
				continue
			}
			outerRing := rings[0]
			for _, seg := range routeSegments {
				for i := 0; i < len(outerRing)-1; i++ {
					if segmentsIntersect(seg[0], seg[1], outerRing[i], outerRing[i+1]) {
						return true
					}
				}
			}
			pts := extractPointList(routeGeom["coordinates"])
			for _, pt := range pts {
				if pointInPolygon(pt, outerRing) {
					return true
				}
			}
		}

	case "LineString":
		refPts := extractPointList(refGeom["coordinates"])
		for _, seg1 := range routeSegments {
			for i := 0; i < len(refPts)-1; i++ {
				if segmentsIntersect(seg1[0], seg1[1], refPts[i], refPts[i+1]) {
					return true
				}
			}
		}

	case "Point":
		pt := extractSinglePoint(refGeom["coordinates"])
		for _, seg := range routeSegments {
			if pointNearSegment(pt, seg[0], seg[1], 0.0001) { // ~10m threshold
				return true
			}
		}
	}

	return false
}

// segmentsIntersect checks whether line segment (p1→p2) intersects (p3→p4)
// using the cross-product method.
func segmentsIntersect(p1, p2, p3, p4 [2]float64) bool {
	d1 := cross(p3, p4, p1)
	d2 := cross(p3, p4, p2)
	d3 := cross(p1, p2, p3)
	d4 := cross(p1, p2, p4)

	if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
		((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) {
		return true
	}

	// Collinear cases
	if d1 == 0 && onSegment(p3, p4, p1) {
		return true
	}
	if d2 == 0 && onSegment(p3, p4, p2) {
		return true
	}
	if d3 == 0 && onSegment(p1, p2, p3) {
		return true
	}
	if d4 == 0 && onSegment(p1, p2, p4) {
		return true
	}

	return false
}

func cross(o, a, b [2]float64) float64 {
	return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
}

func onSegment(p, q, r [2]float64) bool {
	return math.Min(p[0], q[0]) <= r[0] && r[0] <= math.Max(p[0], q[0]) &&
		math.Min(p[1], q[1]) <= r[1] && r[1] <= math.Max(p[1], q[1])
}

// pointInPolygon uses the ray casting algorithm.
func pointInPolygon(pt [2]float64, ring [][2]float64) bool {
	inside := false
	n := len(ring)
	j := n - 1
	for i := 0; i < n; i++ {
		xi, yi := ring[i][0], ring[i][1]
		xj, yj := ring[j][0], ring[j][1]
		if ((yi > pt[1]) != (yj > pt[1])) &&
			(pt[0] < (xj-xi)*(pt[1]-yi)/(yj-yi)+xi) {
			inside = !inside
		}
		j = i
	}
	return inside
}

// pointNearSegment checks if a point is within `threshold` degrees of a segment.
func pointNearSegment(pt, a, b [2]float64, threshold float64) bool {
	dx := b[0] - a[0]
	dy := b[1] - a[1]
	if dx == 0 && dy == 0 {
		return math.Hypot(pt[0]-a[0], pt[1]-a[1]) < threshold
	}
	t := ((pt[0]-a[0])*dx + (pt[1]-a[1])*dy) / (dx*dx + dy*dy)
	t = math.Max(0, math.Min(1, t))
	nearX := a[0] + t*dx
	nearY := a[1] + t*dy
	return math.Hypot(pt[0]-nearX, pt[1]-nearY) < threshold
}

// ─── Coordinate Helpers ───────────────────────────────────────────────────────

func extractGeometry(feature interface{}) map[string]interface{} {
	fm, ok := feature.(map[string]interface{})
	if !ok {
		return nil
	}
	geom, ok := fm["geometry"].(map[string]interface{})
	if !ok {
		return nil
	}
	return geom
}

// extractPointList converts a LineString/ring coordinates array to [][2]float64
func extractPointList(coords interface{}) [][2]float64 {
	arr, ok := coords.([]interface{})
	if !ok {
		return nil
	}
	var result [][2]float64
	for _, item := range arr {
		pair, ok := item.([]interface{})
		if !ok || len(pair) < 2 {
			continue
		}
		x, ok1 := pair[0].(float64)
		y, ok2 := pair[1].(float64)
		if ok1 && ok2 {
			result = append(result, [2]float64{x, y})
		}
	}
	return result
}

// extractLineStrings converts a MultiLineString coordinates array to line point lists.
func extractLineStrings(coords interface{}) [][][2]float64 {
	arr, ok := coords.([]interface{})
	if !ok {
		return nil
	}
	var lines [][][2]float64
	for _, line := range arr {
		points := extractPointList(line)
		if len(points) > 0 {
			lines = append(lines, points)
		}
	}
	return lines
}

// extractRings converts a Polygon coordinates array to rings of points
func extractRings(coords interface{}) [][][2]float64 {
	arr, ok := coords.([]interface{})
	if !ok {
		return nil
	}
	var rings [][][2]float64
	for _, ring := range arr {
		rings = append(rings, extractPointList(ring))
	}
	return rings
}

// extractMultiPolygonRings converts a MultiPolygon coordinates array
func extractMultiPolygonRings(coords interface{}) [][][][2]float64 {
	arr, ok := coords.([]interface{})
	if !ok {
		return nil
	}
	var result [][][][2]float64
	for _, polygon := range arr {
		result = append(result, extractRings(polygon))
	}
	return result
}

// extractSinglePoint converts a Point coordinates array to [2]float64
func extractSinglePoint(coords interface{}) [2]float64 {
	arr, ok := coords.([]interface{})
	if !ok || len(arr) < 2 {
		return [2]float64{}
	}
	x, _ := arr[0].(float64)
	y, _ := arr[1].(float64)
	return [2]float64{x, y}
}

// ─── File Helpers ─────────────────────────────────────────────────────────────

// loadGeoJSONFile loads and caches a GeoJSON file from the mapdata directory
func loadGeoJSONFile(filename string) (interface{}, error) {
	cacheMutex.RLock()
	if data, exists := geoJSONCache[filename]; exists {
		cacheMutex.RUnlock()
		return data, nil
	}
	cacheMutex.RUnlock()

	fullPath := filepath.Join("mapdata", filename)
	raw, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, err
	}

	var parsed interface{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}

	cacheMutex.Lock()
	geoJSONCache[filename] = parsed
	cacheMutex.Unlock()

	return parsed, nil
}

// getGeoJSONFilesInPipeline returns all .geojson files in the mapdata directory
func getGeoJSONFilesInPipeline() ([]string, error) {
	wd, _ := os.Getwd()
	log.Printf("[Navigation] 工作目錄：%s", wd)

	pattern := filepath.Join("mapdata", "*.geojson")
	log.Printf("[Navigation] 搜尋目錄：%s", pattern)

	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	var result []string
	for _, file := range files {
		name := filepath.Base(file)
		log.Printf("[Navigation] 找到檔案：%s", name)
		result = append(result, name)
	}

	log.Printf("[Navigation] 共找到 %d 個檔案", len(result))
	return result, nil
}

// extractFeaturesFromGeoJSON extracts the features array from parsed GeoJSON data
func extractFeaturesFromGeoJSON(data interface{}) ([]interface{}, error) {
	geoJSON, ok := data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid GeoJSON structure")
	}
	features, ok := geoJSON["features"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("no features found in GeoJSON")
	}
	return features, nil
}

// ─── Bounding Box ─────────────────────────────────────────────────────────────

func boundingBoxesOverlap(coords1, coords2 interface{}) bool {
	bbox1 := calculateBoundingBox(coords1)
	bbox2 := calculateBoundingBox(coords2)
	if len(bbox1) != 4 || len(bbox2) != 4 {
		return false
	}
	if bbox1[2] < bbox2[0] || bbox2[2] < bbox1[0] {
		return false
	}
	if bbox1[3] < bbox2[1] || bbox2[3] < bbox1[1] {
		return false
	}
	return true
}

func calculateBoundingBox(coords interface{}) []float64 {
	var minX, minY, maxX, maxY float64
	initialized := false

	var processCoords func(interface{})
	processCoords = func(c interface{}) {
		switch v := c.(type) {
		case []interface{}:
			if len(v) == 2 {
				x, ok1 := v[0].(float64)
				y, ok2 := v[1].(float64)
				if ok1 && ok2 {
					if !initialized {
						minX, maxX, minY, maxY = x, x, y, y
						initialized = true
					} else {
						if x < minX {
							minX = x
						}
						if x > maxX {
							maxX = x
						}
						if y < minY {
							minY = y
						}
						if y > maxY {
							maxY = y
						}
					}
				}
			} else {
				for _, item := range v {
					processCoords(item)
				}
			}
		case map[string]interface{}:
			for _, value := range v {
				processCoords(value)
			}
		}
	}

	processCoords(coords)
	if !initialized {
		return []float64{0, 0, 0, 0}
	}
	return []float64{minX, minY, maxX, maxY}
}

// extractCoordinates recursively extracts coordinate pairs from nested GeoJSON structures
func extractCoordinates(coord interface{}) [][]float64 {
	var result [][]float64
	switch v := coord.(type) {
	case []interface{}:
		if len(v) == 2 {
			if lat, ok1 := v[0].(float64); ok1 {
				if lng, ok2 := v[1].(float64); ok2 {
					result = append(result, []float64{lat, lng})
				}
			}
		} else {
			for _, item := range v {
				result = append(result, extractCoordinates(item)...)
			}
		}
	case map[string]interface{}:
		for _, value := range v {
			result = append(result, extractCoordinates(value)...)
		}
	}
	return result
}
