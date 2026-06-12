/**
 * AnnotationCanvas - Core annotation canvas using Fabric.js
 *
 * Handles image display, zoom/pan, shape rendering, and drawing interaction.
 */

import { useEffect, useRef } from "react";
import * as fabric from "fabric";
import { useProjectStore } from "../../stores/projectStore";
import { useAnnotationStore, debouncedSave } from "../../stores/annotationStore";
import type { Shape } from "../../types/shape";
import { getImageUrl } from "../../api/client";

/** Map shape to Fabric objects */
function shapeToFabric(shape: Shape): fabric.Object[] {
  const objects: fabric.Object[] = [];
  const color = getShapeColor(shape.shape_type);
  const strokeColor = color;
  const fillColor = color.replace(")", ", 0.15)").replace("rgb(", "rgba(");

  const commonProps = {
    stroke: strokeColor,
    strokeWidth: 2,
    fill: fillColor,
    selectable: true,
    hasControls: true,
    hasBorders: true,
    data: { shapeIndex: -1, shapeType: shape.shape_type },
  };

  if (shape.shape_type === "rectangle" && shape.points.length === 4) {
    const [x1, y1] = shape.points[0];
    const [x2, y2] = shape.points[2];
    objects.push(
      new fabric.Rect({
        ...commonProps,
        left: Math.min(x1, x2),
        top: Math.min(y1, y2),
        width: Math.abs(x2 - x1),
        height: Math.abs(y2 - y1),
      })
    );
  } else if (shape.shape_type === "polygon" && shape.points.length >= 3) {
    const pts = shape.points.map((p) => new fabric.Point(p[0], p[1]));
    objects.push(new fabric.Polygon(pts, commonProps));
  } else if (shape.shape_type === "circle" && shape.points.length === 2) {
    const [cx, cy] = shape.points[0];
    const [ex, ey] = shape.points[1];
    const r = Math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2);
    objects.push(
      new fabric.Circle({ ...commonProps, left: cx - r, top: cy - r, radius: r })
    );
  } else if (shape.shape_type === "point" && shape.points.length === 1) {
    const [x, y] = shape.points[0];
    objects.push(
      new fabric.Circle({
        ...commonProps,
        left: x - 4,
        top: y - 4,
        radius: 4,
        fill: strokeColor,
      })
    );
  } else if (shape.shape_type === "line" && shape.points.length === 2) {
    objects.push(
      new fabric.Line(
        [shape.points[0][0], shape.points[0][1], shape.points[1][0], shape.points[1][1]],
        { ...commonProps, fill: undefined }
      )
    );
  } else if (shape.points.length >= 2) {
    // Fallback: polygon for any multi-point shape
    const pts = shape.points.map((p) => new fabric.Point(p[0], p[1]));
    objects.push(new fabric.Polygon(pts, commonProps));
  }

  return objects;
}

function getShapeColor(type: string): string {
  const colors: Record<string, string> = {
    rectangle: "rgb(0, 120, 215)",
    polygon: "rgb(16, 124, 16)",
    point: "rgb(210, 102, 14)",
    line: "rgb(104, 33, 122)",
    circle: "rgb(0, 153, 188)",
    rotation: "rgb(194, 24, 91)",
    quadrilateral: "rgb(63, 81, 181)",
    linestrip: "rgb(239, 108, 0)",
    cuboid: "rgb(255, 171, 0)",
  };
  return colors[type] || "rgb(0, 120, 215)";
}

export default function AnnotationCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<fabric.Canvas | null>(null);
  const drawStateRef = useRef<{
    isDrawing: boolean;
    startPoint: { x: number; y: number } | null;
    currentShape: fabric.Object | null;
    points: { x: number; y: number }[];
  }>({ isDrawing: false, startPoint: null, currentShape: null, points: [] });

  const { currentProject, currentImage } = useProjectStore();
  const { annotation, drawMode, addShape, selectShape, loadAnnotation } =
    useAnnotationStore();

  // Initialize Fabric canvas
  useEffect(() => {
    if (!canvasRef.current || fabricRef.current) return;

    const canvas = new fabric.Canvas(canvasRef.current, {
      selection: true,
      preserveObjectStacking: true,
      backgroundColor: "#1e1e1e",
    });

    // Zoom with mouse wheel
    canvas.on("mouse:wheel", (opt) => {
      const delta = opt.e.deltaY;
      let zoom = canvas.getZoom();
      zoom *= 0.999 ** delta;
      zoom = Math.min(Math.max(0.05, zoom), 20);
      canvas.zoomToPoint(new fabric.Point(opt.e.offsetX, opt.e.offsetY), zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });

    // Pan with middle mouse button or space+left click
    let isPanning = false;
    let lastPosX = 0;
    let lastPosY = 0;

    canvas.on("mouse:down", (opt) => {
      const e = opt.e as MouseEvent;
      if (e.button === 1 || (e.altKey && e.button === 0)) {
        isPanning = true;
        lastPosX = e.clientX;
        lastPosY = e.clientY;
        canvas.selection = false;
      }
    });

    canvas.on("mouse:move", (opt) => {
      if (isPanning) {
        const e = opt.e as MouseEvent;
        const vpt = canvas.viewportTransform!;
        vpt[4] += e.clientX - lastPosX;
        vpt[5] += e.clientY - lastPosY;
        lastPosX = e.clientX;
        lastPosY = e.clientY;
        canvas.requestRenderAll();
      }
    });

    canvas.on("mouse:up", () => {
      isPanning = false;
      canvas.selection = true;
    });

    // Handle selection for shape editing
    canvas.on("selection:created", (opt) => {
      const obj = opt.selected?.[0];
      if (obj && (obj as any).data?.shapeIndex !== undefined) {
        selectShape((obj as any).data.shapeIndex);
      }
    });

    canvas.on("selection:updated", (opt) => {
      const obj = opt.selected?.[0];
      if (obj && (obj as any).data?.shapeIndex !== undefined) {
        selectShape((obj as any).data.shapeIndex);
      }
    });

    canvas.on("selection:cleared", () => {
      useAnnotationStore.getState().clearSelection();
    });

    fabricRef.current = canvas;

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      const container = canvasRef.current?.parentElement;
      if (container) {
        canvas.setDimensions({
          width: container.clientWidth,
          height: container.clientHeight,
        });
        canvas.renderAll();
      }
    });

    if (canvasRef.current?.parentElement) {
      resizeObserver.observe(canvasRef.current.parentElement);
    }

    return () => {
      resizeObserver.disconnect();
      canvas.dispose();
      fabricRef.current = null;
    };
  }, []);

  // Load image when currentImage changes
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !currentProject || !currentImage) return;

    // Load annotation
    loadAnnotation(currentProject.id, currentImage.id);

    // Load image
    const imgUrl = getImageUrl(currentProject.id, currentImage.id);
    fabric.FabricImage.fromURL(imgUrl, { crossOrigin: "anonymous" }).then(
      (img) => {
        canvas.clear();
        canvas.backgroundColor = "#1e1e1e";

        // Scale image to fit
        const canvasWidth = canvas.getWidth();
        const canvasHeight = canvas.getHeight();
        const scaleX = canvasWidth / (img.width || 1);
        const scaleY = canvasHeight / (img.height || 1);
        const scale = Math.min(scaleX, scaleY, 1);

        img.scale(scale);
        img.set({
          left: (canvasWidth - (img.width || 0) * scale) / 2,
          top: (canvasHeight - (img.height || 0) * scale) / 2,
          selectable: false,
          evented: false,
        });

        canvas.add(img);
        canvas.renderAll();
      }
    ).catch((err) => {
      console.error("Failed to load image:", err);
    });
  }, [currentProject?.id, currentImage?.id]);

  // Render shapes when annotation changes
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !annotation) return;

    // Remove existing shape objects (keep the image)
    const objects = canvas.getObjects();
    const toRemove = objects.filter(
      (obj) => (obj as any).data?.shapeIndex !== undefined
    );
    toRemove.forEach((obj) => canvas.remove(obj));

    // Add shapes
    annotation.shapes.forEach((shape, index) => {
      const fabricObjs = shapeToFabric(shape);
      fabricObjs.forEach((obj) => {
        (obj as any).data = {
          ...(obj as any).data || {},
          shapeIndex: index,
          shapeType: shape.shape_type,
        };
        canvas.add(obj);
      });
    });

    canvas.renderAll();
  }, [annotation?.shapes]);

  // Handle drawing mode
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    if (drawMode === "select") {
      canvas.isDrawingMode = false;
      canvas.selection = true;
      canvas.defaultCursor = "default";
      drawStateRef.current.isDrawing = false;
      return;
    }

    canvas.isDrawingMode = false;
    canvas.selection = false;
    canvas.defaultCursor = "crosshair";

    const handleMouseDown = (opt: any) => {
      if (opt.e.button !== 0 || opt.e.altKey) return;
      const pointer = canvas.getScenePoint(opt.e);
      const state = drawStateRef.current;

      if (drawMode === "rectangle") {
        state.isDrawing = true;
        state.startPoint = { x: pointer.x, y: pointer.y };
        const rect = new fabric.Rect({
          left: pointer.x,
          top: pointer.y,
          width: 0,
          height: 0,
          stroke: "rgb(0, 120, 215)",
          strokeWidth: 2,
          fill: "rgba(0, 120, 215, 0.15)",
          selectable: false,
          evented: false,
        });
        canvas.add(rect);
        state.currentShape = rect;
      } else if (drawMode === "point") {
        addShape({
          label: "label",
          points: [[pointer.x, pointer.y]],
          shape_type: "point",
          flags: {},
        });
        if (currentProject && currentImage) {
          debouncedSave(currentProject.id, currentImage.id);
        }
      }
    };

    const handleMouseMove = (opt: any) => {
      const state = drawStateRef.current;
      if (!state.isDrawing || !state.startPoint) return;

      const pointer = canvas.getScenePoint(opt.e);

      if (drawMode === "rectangle" && state.currentShape) {
        const rect = state.currentShape as fabric.Rect;
        const left = Math.min(state.startPoint.x, pointer.x);
        const top = Math.min(state.startPoint.y, pointer.y);
        const width = Math.abs(pointer.x - state.startPoint.x);
        const height = Math.abs(pointer.y - state.startPoint.y);
        rect.set({ left, top, width, height });
        canvas.renderAll();
      }
    };

    const handleMouseUp = (opt: any) => {
      const state = drawStateRef.current;
      if (!state.isDrawing || !state.startPoint) return;

      const pointer = canvas.getScenePoint(opt.e);

      if (drawMode === "rectangle") {
        // Remove the temp shape
        if (state.currentShape) {
          canvas.remove(state.currentShape);
        }

        const w = Math.abs(pointer.x - state.startPoint.x);
        const h = Math.abs(pointer.y - state.startPoint.y);

        if (w > 3 && h > 3) {
          const x1 = Math.min(state.startPoint.x, pointer.x);
          const y1 = Math.min(state.startPoint.y, pointer.y);
          const x2 = x1 + w;
          const y2 = y1 + h;

          addShape({
            label: "label",
            points: [
              [x1, y1],
              [x2, y1],
              [x2, y2],
              [x1, y2],
            ],
            shape_type: "rectangle",
            flags: {},
          });

          if (currentProject && currentImage) {
            debouncedSave(currentProject.id, currentImage.id);
          }
        }

        state.isDrawing = false;
        state.startPoint = null;
        state.currentShape = null;
      }
    };

    canvas.on("mouse:down", handleMouseDown);
    canvas.on("mouse:move", handleMouseMove);
    canvas.on("mouse:up", handleMouseUp);

    return () => {
      canvas.off("mouse:down", handleMouseDown);
      canvas.off("mouse:move", handleMouseMove);
      canvas.off("mouse:up", handleMouseUp);
    };
  }, [drawMode]);

  return (
    <div
      style={{
        flex: 1,
        position: "relative",
        background: "#1e1e1e",
        overflow: "hidden",
      }}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
