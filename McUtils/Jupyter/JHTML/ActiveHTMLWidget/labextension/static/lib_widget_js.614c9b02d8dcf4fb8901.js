"use strict";
(self["webpackChunkActiveHTMLWidget"] = self["webpackChunkActiveHTMLWidget"] || []).push([["lib_widget_js"],{

/***/ "./lib/version.js":
/*!************************!*\
  !*** ./lib/version.js ***!
  \************************/
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


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
__webpack_require__(/*! bootstrap */ "webpack/sharing/consume/default/bootstrap/bootstrap");
class LayoutManagerWidget extends widgets_1.Widget {
    constructor(options) {
        let view = options.view;
        //@ts-ignore
        options.tag = view.tagName;
        super(options);
        this._view = view;
        this.layout = new widgets_1.PanelLayout({ fitPolicy: 'set-no-constraint' });
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
    // constructDict(listPair:any) {
    //     let res = {};
    //     let keys = listPair[0];
    //     let vals = listPair[1];
    //     for (let i = 0; i < keys.length; i++) {
    //         //@ts-ignore
    //         res[keys[i]] = vals[i];
    //     }
    //     return res;
    // }
    initialize(parameters) {
        super.initialize(parameters);
        //@ts-ignore
        this.children_views = new base_1.ViewList(this.add_child_model, null, this);
        this.listenTo(this.model, 'change:children', this.updateBody);
        this.listenTo(this.model, 'change:innerHTML', this.updateBody);
        this.listenTo(this.model, 'change:textContent', this.updateBody);
        this.listenTo(this.model, 'change:styleDict', this.updateStyles);
        this.listenTo(this.model, 'change:classList', this.updateClassList);
        this.listenTo(this.model, 'change:value', this.updateValue);
        this.listenTo(this.model, 'change:elementAttributes', this.updateAttributes);
        this.listenTo(this.model, 'change:eventPropertiesDict', this.updateEvents);
        this._currentEvents = {};
        this._currentClasses = new Set();
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
    setLayout(layout, oldLayout) { } // null override
    setStyle(style, oldStyle) { } // null override
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
                    // console.log(">>>", prop, elementStyles[prop], typeof prop);
                    this.el.style.setProperty(prop, elementStyles[prop]);
                    // console.log("<<<", prop, this.el.style.getPropertyValue(prop));
                    this._currentStyles.add(prop);
                }
            }
        }
    }
    updateStyles() {
        this.setStyles();
        this.removeStyles();
    }
    setClasses() {
        if (this.model.get("_debugPrint")) {
            console.log(this.el, "Element Classes:", this.model.get("classList"));
        }
        let classList = this.model.get("classList");
        for (let cls of classList) {
            this.el.classList.add(cls);
            this._currentClasses.add(cls);
        }
    }
    removeClasses() {
        if (this.model.get("_debugPrint")) {
            console.log(this.el, "Element Classes:", this.model.get("classList"));
        }
        let current = this._currentClasses;
        let classes = this.model.get("classList");
        for (let prop of current) {
            if (!classes.includes(prop)) {
                this.el.classList.remove(prop);
                this._currentClasses.delete(prop);
            }
        }
    }
    updateClassList() {
        this.setClasses();
        this.removeClasses();
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
        if (debug) {
            console.log(this.el, "Element Properties:", attrs);
        }
        for (let prop in attrs) {
            let val = attrs[prop];
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
        let debug = this.model.get("_debugPrint");
        if (el !== undefined) {
            let is_checkbox = el.getAttribute('type') === 'checkbox' || el.getAttribute('type') === 'radio';
            let multiple = el.getAttribute('multiple');
            if (is_checkbox) {
                let checked = el.checked;
                if (checked !== undefined) {
                    let newVal = this.model.get('value');
                    let checkVal = newVal.length > 0 && newVal != "false" && newVal != "0";
                    if (debug) {
                        console.log('updating checked', checked, "->", checkVal);
                    }
                    if (checkVal !== checked) {
                        el.checked = checkVal;
                    }
                }
            }
            else if (multiple) {
                let el = this.el;
                let opts = el.selectedOptions;
                if (opts !== undefined) {
                    let val = [];
                    for (let i = 0; i < opts.length; i++) {
                        let o = opts[i];
                        val.push(o.value);
                    }
                    let newValStr = this.model.get('value');
                    if (typeof newValStr === 'string') {
                        let testVal = val.join('&&');
                        if (debug) {
                            console.log('updating selection', testVal, "->", newValStr);
                        }
                        if (newValStr !== testVal) {
                            let splitVals = newValStr.split("&&");
                            for (let i = 0; i < el.options.length; i++) {
                                let o = el.options[i];
                                o.selected = (splitVals.indexOf(o.value) > -1);
                                if (debug) {
                                    console.log('updating selection...', o.selected);
                                }
                            }
                        }
                    }
                }
            }
            else {
                let val = el.value;
                if (val !== undefined) {
                    let newVal = this.model.get('value');
                    if (debug) {
                        console.log('updating value', val, "->", newVal);
                    }
                    if (newVal !== val) {
                        el.value = newVal;
                    }
                }
            }
        }
    }
    setEvents() {
        let listeners = this.model.get('eventPropertiesDict');
        let debug = this.model.get("_debugPrint");
        if (debug) {
            console.log(this.el, "Adding Events:", listeners);
        }
        for (let key in listeners) {
            if (listeners.hasOwnProperty(key)) {
                if (!this._currentEvents.hasOwnProperty(key)) {
                    this._currentEvents[key] = [
                        listeners[key],
                        this.constructEventListener(key, listeners[key])
                    ];
                    this.el.addEventListener(key, this._currentEvents[key][1]);
                }
                else if (this._currentEvents[key][0] !== listeners[key]) {
                    this.el.removeEventListener(key, this._currentEvents[key][1]);
                    this._currentEvents[key] = [
                        listeners[key],
                        this.constructEventListener(key, listeners[key])
                    ];
                    this.el.addEventListener(key, this._currentEvents[key][1]);
                }
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
                    this.el.removeEventListener(prop, this._currentEvents[prop][1]);
                    this._currentEvents.delete(prop);
                }
            }
        }
    }
    updateEvents() {
        this.setEvents();
        this.removeEvents();
    }
    render() {
        super.render();
        this.el.classList.remove('lm-Widget', 'p-Widget');
        this.update();
    }
    update() {
        this.updateBody();
        // this.updateTextContent();
        this.updateAttribute('id');
        this.updateAttributes();
        this.updateClassList();
        this.setStyles();
        this.setEvents();
        this.updateValue();
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
        let el = this.el;
        let is_checkbox = el.getAttribute('type') === 'checkbox' || el.getAttribute('type') === 'radio';
        let multiple = el.getAttribute('multiple');
        if (is_checkbox) {
            let checked = target.checked;
            if (checked !== undefined) {
                this.model.set('value', checked ? "true" : "false", { updated_view: this });
                this.touch();
            }
        }
        else if (multiple) {
            let el = this.el;
            let opts = el.selectedOptions;
            if (opts !== undefined) {
                let val = [];
                for (let i = 0; i < opts.length; i++) {
                    let o = opts[i];
                    val.push(o.value);
                }
                let newVal = val.join('&&');
                this.model.set('value', newVal, { updated_view: this });
                this.touch();
            }
        }
        else {
            let val = target.value;
            if (val !== undefined) {
                this.model.set('value', val, { updated_view: this });
                this.touch();
            }
        }
    }
    constructEventListener(eventName, propData) {
        let parent = this;
        return function (e) {
            let props;
            if (Array.isArray(propData)) {
                props = propData;
            }
            else if (propData === undefined || propData === null) {
                props = parent.model.get('defaultEventProperties');
            }
            else {
                //@ts-ignore
                props = propData['fields'];
                //@ts-ignore
                let prop = propData['propagate'];
                if (prop !== true) {
                    e.stopPropagation();
                }
            }
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

/***/ "./package.json":
/*!**********************!*\
  !*** ./package.json ***!
  \**********************/
/***/ ((module) => {

module.exports = JSON.parse('{"name":"ActiveHTMLWidget","version":"0.1.0","description":"A Custom Jupyter Widget Library","keywords":["jupyter","jupyterlab","jupyterlab-extension","widgets"],"files":["lib/**/*.js","dist/*.js","css/*.css"],"homepage":"https://github.com//ActiveHTMLWidget","bugs":{"url":"https://github.com//ActiveHTMLWidget/issues"},"license":"MIT","author":{"name":"b3m2a1","email":"b3m2a1@gmail.com"},"main":"lib/index.js","types":"./lib/index.d.ts","style":"css/index.css","sideEffects":["css/*.css"],"repository":{"type":"git","url":"https://github.com//ActiveHTMLWidget"},"scripts":{"build":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension:dev","build:prod":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension","build:labextension":"jupyter labextension build .","build:labextension:dev":"jupyter labextension build --development True .","build:lib":"tsc","build:nbextension":"webpack","clean":"yarn run clean:lib && yarn run clean:nbextension && yarn run clean:labextension","clean:lib":"rimraf lib","clean:labextension":"rimraf ActiveHTMLWidget/labextension","clean:nbextension":"rimraf ActiveHTMLWidget/nbextension/static/index.js","lint":"eslint . --ext .ts,.tsx --fix","lint:check":"eslint . --ext .ts,.tsx","prepack":"yarn run build:lib","test":"jest","watch":"npm-run-all -p watch:*","watch:lib":"tsc -w","watch:nbextension":"webpack --watch --mode=development","watch:labextension":"jupyter labextension watch ."},"dependencies":{"@jupyter-widgets/base":"^1.1.10 || ^2.0.0 || ^3.0.0 || ^4.0.0","bootstrap":"^5.1.3","sass":"^1.50.1"},"devDependencies":{"@babel/core":"^7.5.0","@babel/preset-env":"^7.5.0","@jupyterlab/builder":"^3.0.0","@phosphor/application":"^1.6.0","@phosphor/widgets":"^1.6.0","@types/jest":"^26.0.0","@types/webpack-env":"^1.13.6","@typescript-eslint/eslint-plugin":"^3.6.0","@typescript-eslint/parser":"^3.6.0","acorn":"^7.2.0","css-loader":"^3.2.0","eslint":"^7.4.0","eslint-config-prettier":"^6.11.0","eslint-plugin-prettier":"^3.1.4","fs-extra":"^7.0.0","identity-obj-proxy":"^3.0.0","jest":"^26.0.0","mkdirp":"^0.5.1","npm-run-all":"^4.1.3","prettier":"^2.0.5","rimraf":"^2.6.2","source-map-loader":"^1.1.3","style-loader":"^1.0.0","ts-jest":"^26.0.0","ts-loader":"^8.0.0","typescript":"~4.1.3","webpack":"^5.0.0","webpack-cli":"^4.0.0"},"jupyterlab":{"extension":"lib/plugin","outputDir":"ActiveHTMLWidget/labextension/","sharedPackages":{"@jupyter-widgets/base":{"bundled":false,"singleton":true}}}}');

/***/ })

}]);
//# sourceMappingURL=lib_widget_js.614c9b02d8dcf4fb8901.js.map