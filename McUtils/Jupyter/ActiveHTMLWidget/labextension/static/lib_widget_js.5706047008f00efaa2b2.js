(self["webpackChunkActiveHTMLWidget"] = self["webpackChunkActiveHTMLWidget"] || []).push([["lib_widget_js"],{

/***/ "./lib/version.js":
/*!************************!*\
  !*** ./lib/version.js ***!
  \************************/
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {

"use strict";

// Copyright (c) b3m2a1
// Distributed under the terms of the Modified BSD License.
Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.MODULE_NAME = exports.MODULE_VERSION = void 0;
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
// eslint-disable-next-line @typescript-eslint/no-var-requires
const data = __webpack_require__(/*! ../package.json */ "./package.json");
/**
 * The _model_module_version/_view_module_version this package implements.
 *
 * The html widget manager assumes that this is the same as the npm package
 * version number.
 */
exports.MODULE_VERSION = data.version;
/*
 * The current package name.
 */
exports.MODULE_NAME = data.name;
//# sourceMappingURL=version.js.map

/***/ }),

/***/ "./lib/widget.js":
/*!***********************!*\
  !*** ./lib/widget.js ***!
  \***********************/
/***/ (function(__unused_webpack_module, exports, __webpack_require__) {

"use strict";

// Copyright (c) b3m2a1
// Distributed under the terms of the Modified BSD License.
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.ActiveHTMLView = exports.ActiveHTMLModel = void 0;
const base_1 = __webpack_require__(/*! @jupyter-widgets/base */ "webpack/sharing/consume/default/@jupyter-widgets/base");
const version_1 = __webpack_require__(/*! ./version */ "./lib/version.js");
const widgets_1 = __webpack_require__(/*! @lumino/widgets */ "webpack/sharing/consume/default/@lumino/widgets");
const algorithm_1 = __webpack_require__(/*! @lumino/algorithm */ "webpack/sharing/consume/default/@lumino/algorithm");
const messaging_1 = __webpack_require__(/*! @lumino/messaging */ "webpack/sharing/consume/default/@lumino/messaging");
const jquery_1 = __importDefault(__webpack_require__(/*! jquery */ "./node_modules/jquery/dist/jquery.js"));
// Import the CSS
__webpack_require__(/*! ../css/widget.css */ "./css/widget.css");
class LayoutManagerWidget extends widgets_1.Widget {
    constructor(options) {
        let view = options.view;
        //@ts-ignore
        options.tag = view.tagName;
        super(options);
        this._view = view;
        this.layout = new widgets_1.PanelLayout();
    }
    dispose() {
        if (this.isDisposed) {
            return;
        }
        super.dispose();
        if (this._view) {
            this._view.remove();
        }
        //@ts-ignore
        this._view = null;
    }
    processMessage(msg) {
        super.processMessage(msg);
        this._view.processPhosphorMessage(msg);
    }
    get widgets() {
        return this.layout.widgets;
    }
    addWidget(widget) {
        this.layout.addWidget(widget);
    }
    insertWidget(index, widget) {
        this.layout.insertWidget(index, widget);
    }
}
class ActiveHTMLModel extends base_1.DOMWidgetModel {
    defaults() {
        return Object.assign(Object.assign({}, super.defaults()), { _model_name: ActiveHTMLModel.model_name, _model_module: ActiveHTMLModel.model_module, _model_module_version: ActiveHTMLModel.model_module_version, _view_name: ActiveHTMLModel.view_name, _view_module: ActiveHTMLModel.view_module, _view_module_version: ActiveHTMLModel.view_module_version, tagName: 'div', children: [], classList: [], innerHTML: "", textContent: "", _bodyType: "", _debugPrint: false, styleDict: {}, elementAttributes: {}, id: "", value: "", trackInput: false, continuousUpdate: true, eventPropertiesDict: {}, defaultEventProperties: [
                "bubbles", "cancelable", "composed",
                "target", "timestamp", "type",
                "key", "repeat",
                "button", "buttons",
                "alKey", "shiftKey", "ctrlKey", "metaKey"
            ] });
    }
}
exports.ActiveHTMLModel = ActiveHTMLModel;
ActiveHTMLModel.serializers = Object.assign(Object.assign({}, base_1.DOMWidgetModel.serializers), { 
    // Add any extra serializers here
    //@ts-ignore
    children: { deserialize: base_1.unpack_models } });
ActiveHTMLModel.model_name = 'ActiveHTMLModel';
ActiveHTMLModel.model_module = version_1.MODULE_NAME;
ActiveHTMLModel.model_module_version = version_1.MODULE_VERSION;
ActiveHTMLModel.view_name = 'ActiveHTMLView'; // Set to null if no view
ActiveHTMLModel.view_module = version_1.MODULE_NAME; // Set to null if no view
ActiveHTMLModel.view_module_version = version_1.MODULE_VERSION;
class ActiveHTMLView extends base_1.DOMWidgetView {
    initialize(parameters) {
        super.initialize(parameters);
        //@ts-ignore
        this.children_views = new base_1.ViewList(this.add_child_model, null, this);
        this.listenTo(this.model, 'change:children', this.updateBody);
        this.listenTo(this.model, 'change:innerHTML', this.updateBody);
        this.listenTo(this.model, 'change:textContent', this.updateBody);
        this._currentEvents = {};
        this._currentStyles = new Set();
    }
    removeStyles() {
        let newStyles = this.model.get("styleDict");
        let current = this._currentStyles;
        for (let prop of current) {
            if (!newStyles.hasOwnProperty(prop)) {
                this.el.style.removeProperty(prop);
                this._currentStyles.delete(prop);
            }
        }
    }
    setStyles() {
        let elementStyles = this.model.get("styleDict");
        if (elementStyles.length === 0) {
            this._currentStyles.clear();
            this.el.removeAttribute('style');
        }
        else {
            if (this.model.get("_debugPrint")) {
                console.log(this.el, "Element Styles:", elementStyles);
            }
            for (let prop in elementStyles) {
                if (elementStyles.hasOwnProperty(prop)) {
                    // console.log(">", prop, elementStyles[prop]);
                    this.el.style.setProperty(prop, elementStyles[prop]);
                    this._currentStyles.add(prop);
                }
            }
        }
    }
    updateStyles(model, value, options) {
        this.setStyles();
        this.removeStyles();
    }
    // Manage classes
    updateClassList() {
        // @ts-ignore
        for (let cls of this.el.classList) {
            this.el.classList.remove(cls);
        }
        for (let cls of this.model.get("classList")) {
            this.el.classList.add(cls);
        }
    }
    //manage body of element (borrowed from ipywidgets.Box)
    _createElement(tagName) {
        this.pWidget = new LayoutManagerWidget({ view: this });
        return this.pWidget.node;
    }
    _setElement(el) {
        if (this.el || el !== this.pWidget.node) {
            // Boxes don't allow setting the element beyond the initial creation.
            throw new Error('Cannot reset the DOM element.');
        }
        this.el = this.pWidget.node;
        this.$el = jquery_1.default(this.pWidget.node);
    }
    update_children() {
        if (this.children_views !== null) {
            this.children_views.update(this.model.get('children')).then((views) => {
                // Notify all children that their sizes may have changed.
                views.forEach((view) => {
                    messaging_1.MessageLoop.postMessage(view.pWidget, widgets_1.Widget.ResizeMessage.UnknownSize);
                });
            });
        }
    }
    add_child_model(model) {
        // we insert a dummy element so the order is preserved when we add
        // the rendered content later.
        let dummy = new widgets_1.Widget();
        //@ts-ignore
        this.pWidget.addWidget(dummy);
        return this.create_child_view(model).then((view) => {
            // replace the dummy widget with the new one.
            //@ts-ignore
            let i = algorithm_1.ArrayExt.firstIndexOf(this.pWidget.widgets, dummy);
            //@ts-ignore
            this.pWidget.insertWidget(i, view.pWidget);
            dummy.dispose();
            return view;
        }).catch(base_1.reject('Could not add child view to box', true));
    }
    remove() {
        this.children_views = null;
        super.remove();
    }
    updateBody() {
        let children = this.model.get('children');
        let debug = this.model.get("_debugPrint");
        if (children.length > 0) {
            if (debug) {
                console.log(this.el, "Updating Children...");
            }
            this.update_children();
            // console.log('for the future...');
            // this.updateChildren();
        }
        else {
            let html = this.model.get("innerHTML");
            if (html.length > 0) {
                if (debug) {
                    console.log(this.el, "Updating HTML...");
                }
                this.updateInnerHTML();
            }
            else {
                let text = this.model.get("textContent");
                if (text.length > 0) {
                    if (debug) {
                        console.log(this.el, "Updating Text...");
                    }
                    this.updateTextContent();
                }
                else {
                    if (debug) {
                        console.log(this.el, "Updating HTML...");
                    }
                    this.updateInnerHTML();
                }
            }
        }
    }
    updateInnerHTML() {
        // let bodyType = this.model.get('_bodyType');
        // if (bodyType !== "html") {
        //   this.resetBody();
        // }
        let val = this.model.get("innerHTML");
        let cur = this.el.innerHTML;
        if (val !== cur) {
            this.el.innerHTML = val;
        }
        // if (bodyType !== "html") {
        //   this.model.set('_bodyType', "html");
        // }
    }
    updateTextContent() {
        // let bodyType = this.model.get('_bodyType');
        // if (bodyType !== "html") {
        //   this.resetBody(bodyType);
        // }
        let val = this.model.get("textContent");
        let cur = this.el.textContent;
        if (val !== cur) {
            this.el.textContent = val;
        }
        // if (bodyType !== "html") {
        //   this.model.set('_bodyType', "html");
        // }
    }
    // Setting attributes (like id)
    updateAttribute(attrName) {
        let val = this.model.get(attrName);
        if (val === "") {
            this.el.removeAttribute(attrName);
        }
        else {
            this.el.setAttribute(attrName, val);
        }
    }
    updateAttributeFromQuery(attrName, queryName) {
        let val = this.model.get(queryName);
        if (val === "") {
            this.el.removeAttribute(attrName);
        }
        else {
            this.el.setAttribute(attrName, val);
        }
    }
    updateAttributes() {
        let attrs = this.model.get('elementAttributes');
        let debug = this.model.get("_debugPrint");
        for (let prop in attrs) {
            let val = attrs[prop];
            if (debug) {
                console.log(this.el, "Adding Property:", prop);
            }
            if (val === "") {
                this.el.removeAttribute(prop);
            }
            else {
                this.el.setAttribute(prop, val);
            }
        }
    }
    updateValue() {
        let el = this.el;
        if (el !== undefined) {
            let val = el.value;
            if (val !== undefined) {
                let newVal = this.model.get('value');
                if (newVal !== val) {
                    el.value = newVal;
                }
            }
        }
    }
    setEvents() {
        let listeners = this.model.get('eventPropertiesDict');
        let debug = this.model.get("_debugPrint");
        for (let key in listeners) {
            if (listeners.hasOwnProperty(key)) {
                if (debug) {
                    console.log(this.el, "Adding Event:", key);
                }
                this._currentEvents[key] = this.constructEventListener(key, listeners[key]);
                this.el.addEventListener(key, this._currentEvents[key]);
            }
        }
    }
    removeEvents() {
        let newListeners = this.model.get('eventPropertiesDict');
        let current = this._currentEvents;
        let debug = this.model.get("_debugPrint");
        for (let prop in current) {
            if (current.hasOwnProperty(prop)) {
                if (!newListeners.hasOwnProperty(prop)) {
                    if (debug) {
                        console.log(this.el, "Removing Event:", prop);
                    }
                    this.el.removeEventListener(prop, this._currentEvents[prop]);
                    this._currentEvents.delete(prop);
                }
            }
        }
    }
    updateEvents() {
        this.setEvents();
        this.removeEvents();
    }
    // removeEvents() {
    //     let listeners = this.model.get('eventPropertiesDict') as Record<string, string[]>;
    //     for (let key in listeners) {
    //         if (listeners.hasOwnProperty(key)) {
    //             this.el.addEventListener(key, this.constructEventListener(key, listeners[key]));
    //             this._currentEvents.add(key);
    //         }
    //     }
    //     // console.log(events);
    // }
    render() {
        super.render();
        this.update();
        this.model.on('change:style', this.updateStyles, this);
        this.model.on('change:classList', this.updateClassList, this);
        this.model.on('change:value', this.updateValue, this);
        this.model.on('change:eventPropertiesDict', this.updateEvents, this);
    }
    update() {
        this.updateBody();
        // this.updateTextContent();
        this.updateAttribute('id');
        this.updateValue();
        this.updateAttributes();
        this.updateClassList();
        this.setStyles();
        this.setEvents();
        // this.el.classList = this.model.get("classList");
    }
    // @ts-ignore
    get tagName() {
        // We can't make this an attribute with a default value
        // since it would be set after it is needed in the
        // constructor.
        return this.model.get('tagName');
    }
    // Adapted from the "TextView" from the core package
    events() {
        let events = {};
        if (this.model.get('trackInput')) {
            // let tagName = this.model.get('tagName');
            let key = 'keydown'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleKeyDown';
            key = 'keypress'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleKeypress';
            key = 'input'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleChanging';
            key = 'change'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleChanged';
        }
        return events;
    }
    handleKeyDown(e) {
        e.stopPropagation();
    }
    handleKeypress(e) {
        e.stopPropagation();
    }
    handleChanging(e) {
        if (this.model.get('continuousUpdate')) {
            this.handleChanged(e);
        }
    }
    handleChanged(e) {
        let target = e.target;
        let val = target.value;
        if (val !== undefined) {
            this.model.set('value', val, { updated_view: this });
            this.touch();
        }
    }
    constructEventListener(eventName, props) {
        let parent = this;
        return function (e) {
            let debug = parent.model.get('_debugPrint');
            if (debug) {
                console.log(parent.el, "Handling event:", eventName);
            }
            // console.log("|", eventName, props);
            parent.sendEventMessage(e, parent.constructEventMessage(e, props, eventName));
        };
    }
    constructEventMessage(e, props, eventName) {
        if (props === undefined || props === null) {
            props = this.model.get('defaultEventProperties');
        }
        if (props === undefined) {
            props = ['target'];
        }
        let eventMessage = {};
        if (eventName !== undefined) {
            eventMessage['eventName'] = eventName;
        }
        for (let p of props) {
            // @ts-ignore
            let val = e[p];
            if (p === "target") {
                val = {};
                let t = e.target;
                val['tag'] = t.tagName;
                val['innerHTML'] = t.innerHTML;
                for (let p of t.getAttributeNames()) {
                    val[p] = t.getAttribute(p);
                }
            }
            eventMessage[p] = val;
        }
        return eventMessage;
    }
    sendEventMessage(e, message) {
        e.stopPropagation();
        if (message === undefined) {
            message = this.constructEventMessage(e);
        }
        let debug = this.model.get('_debugPrint');
        if (debug) {
            console.log(this.el, "Sending message:", message);
        }
        this.send(message);
    }
}
exports.ActiveHTMLView = ActiveHTMLView;
//# sourceMappingURL=widget.js.map

/***/ }),

/***/ "./node_modules/css-loader/dist/cjs.js!./css/widget.css":
/*!**************************************************************!*\
  !*** ./node_modules/css-loader/dist/cjs.js!./css/widget.css ***!
  \**************************************************************/
/***/ ((module, exports, __webpack_require__) => {

// Imports
var ___CSS_LOADER_API_IMPORT___ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/api.js */ "./node_modules/css-loader/dist/runtime/api.js");
exports = ___CSS_LOADER_API_IMPORT___(false);
// Module
exports.push([module.id, ".custom-widget {\n  background-color: lightseagreen;\n  padding: 0px 2px;\n}\n", ""]);
// Exports
module.exports = exports;


/***/ }),

/***/ "./css/widget.css":
/*!************************!*\
  !*** ./css/widget.css ***!
  \************************/
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {

var api = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js */ "./node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js");
            var content = __webpack_require__(/*! !!../node_modules/css-loader/dist/cjs.js!./widget.css */ "./node_modules/css-loader/dist/cjs.js!./css/widget.css");

            content = content.__esModule ? content.default : content;

            if (typeof content === 'string') {
              content = [[module.id, content, '']];
            }

var options = {};

options.insert = "head";
options.singleton = false;

var update = api(content, options);



module.exports = content.locals || {};

/***/ }),

/***/ "./package.json":
/*!**********************!*\
  !*** ./package.json ***!
  \**********************/
/***/ ((module) => {

"use strict";
module.exports = JSON.parse('{"name":"ActiveHTMLWidget","version":"0.1.0","description":"A Custom Jupyter Widget Library","keywords":["jupyter","jupyterlab","jupyterlab-extension","widgets"],"files":["lib/**/*.js","dist/*.js","css/*.css"],"homepage":"https://github.com//ActiveHTMLWidget","bugs":{"url":"https://github.com//ActiveHTMLWidget/issues"},"license":"BSD-3-Clause","author":{"name":"b3m2a1","email":"b3m2a1@gmail.com"},"main":"lib/index.js","types":"./lib/index.d.ts","repository":{"type":"git","url":"https://github.com//ActiveHTMLWidget"},"scripts":{"build":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension:dev","build:prod":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension","build:labextension":"jupyter labextension build .","build:labextension:dev":"jupyter labextension build --development True .","build:lib":"tsc","build:nbextension":"webpack","clean":"yarn run clean:lib && yarn run clean:nbextension && yarn run clean:labextension","clean:lib":"rimraf lib","clean:labextension":"rimraf ActiveHTMLWidget/labextension","clean:nbextension":"rimraf ActiveHTMLWidget/nbextension/static/index.js","lint":"eslint . --ext .ts,.tsx --fix","lint:check":"eslint . --ext .ts,.tsx","prepack":"yarn run build:lib","test":"jest","watch":"npm-run-all -p watch:*","watch:lib":"tsc -w","watch:nbextension":"webpack --watch --mode=development","watch:labextension":"jupyter labextension watch ."},"dependencies":{"@jupyter-widgets/base":"^1.1.10 || ^2.0.0 || ^3.0.0 || ^4.0.0"},"devDependencies":{"@babel/core":"^7.5.0","@babel/preset-env":"^7.5.0","@jupyterlab/builder":"^3.0.0","@phosphor/application":"^1.6.0","@phosphor/widgets":"^1.6.0","@types/jest":"^26.0.0","@types/webpack-env":"^1.13.6","@typescript-eslint/eslint-plugin":"^3.6.0","@typescript-eslint/parser":"^3.6.0","acorn":"^7.2.0","css-loader":"^3.2.0","eslint":"^7.4.0","eslint-config-prettier":"^6.11.0","eslint-plugin-prettier":"^3.1.4","fs-extra":"^7.0.0","identity-obj-proxy":"^3.0.0","jest":"^26.0.0","mkdirp":"^0.5.1","npm-run-all":"^4.1.3","prettier":"^2.0.5","rimraf":"^2.6.2","source-map-loader":"^1.1.3","style-loader":"^1.0.0","ts-jest":"^26.0.0","ts-loader":"^8.0.0","typescript":"~4.1.3","webpack":"^5.0.0","webpack-cli":"^4.0.0"},"jupyterlab":{"extension":"lib/plugin","outputDir":"ActiveHTMLWidget/labextension/","sharedPackages":{"@jupyter-widgets/base":{"bundled":false,"singleton":true}}}}');

/***/ })

}]);
//# sourceMappingURL=lib_widget_js.5706047008f00efaa2b2.js.map